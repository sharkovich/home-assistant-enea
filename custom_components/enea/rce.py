"""Hourly RCE prices (rynkowa cena energii) for a prosumer's net-billing.

Under net-billing each kWh returned to the grid goes into the prosumer deposit
at the RCE of the hour it was returned in (a negative price counts as zero),
multiplied by NET_BILLING_DEPOSIT_FACTOR.  PSE publishes RCE in its public
report API (rce-pln): one row per 15-minute period since PSE moved to
quarter-hour settlement, one per hour before.  hourly_prices averages a hour's
periods into the price of that hour.

RceCache keeps the prices of days already fetched in a Store, so repricing the
whole history on every refresh never downloads it again: past prices do not
change.  A day without prices is asked for again until it is older than
RCE_MISSING_DAY_GRACE_DAYS — PSE publishes a day's prices the day before, but
the API may lag, and a day before RCE existed will never have any.
"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any, Protocol

import aiohttp

from .const import RCE_MISSING_DAY_GRACE_DAYS, RCE_PAGE_SIZE, URL_RCE

_LOGGER = logging.getLogger(__name__)

type RceFetch = Callable[[date, date], Awaitable[list[dict[str, Any]]]]


class RceError(Exception):
    """The RCE prices could not be fetched from PSE."""


class _Store(Protocol):
    """The part of homeassistant.helpers.storage.Store the cache uses."""

    async def async_load(self) -> dict[str, Any] | None:
        """Return the stored data, None when nothing was stored."""
        ...

    async def async_save(self, data: dict[str, Any]) -> None:
        """Store data."""
        ...


def _minutes(clock: str) -> int:
    """Return the minutes since midnight of an "HH:MM" clock time."""
    hours, minutes = clock.strip().split(":")
    return int(hours) * 60 + int(minutes)


def hourly_prices(rows: list[dict[str, Any]]) -> dict[int, float]:
    """Return the RCE of each hour in PLN/kWh, keyed by the UTC timestamp of its start.

    A row's period ends at dtime_utc and lasts as long as period_utc says
    ("10:00 - 10:15", "23:45 - 00:00", "12:00 - 13:00"); every period counts
    towards the hour it starts in, and the hour's price is their mean.
    """
    sums: dict[int, list[float]] = {}
    for row in rows:
        price = row.get("rce_pln")
        if price is None:
            continue
        end = datetime.strptime(row["dtime_utc"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
        begin_clock, end_clock = row["period_utc"].split("-")
        length = (_minutes(end_clock) - _minutes(begin_clock)) % (24 * 60) or 24 * 60
        start = end - timedelta(minutes=length)
        hour = int(start.replace(minute=0, second=0, microsecond=0).timestamp())
        total = sums.setdefault(hour, [0.0, 0.0])
        total[0] += float(price) / 1000
        total[1] += 1
    return {hour: total / count for hour, (total, count) in sums.items()}


async def async_fetch_rows(session: aiohttp.ClientSession, first: date, last: date) -> list[dict[str, Any]]:
    """Return the rce-pln report rows of the business days from first to last.

    Follows the report's nextLink until the whole range is read.

    Raises:
        RceError: The API could not be reached or did not answer with data.
    """
    url: str | None = URL_RCE
    params: dict[str, str] | None = {
        "$filter": f"business_date ge '{first.isoformat()}' and business_date le '{last.isoformat()}'",
        "$select": "dtime_utc,period_utc,rce_pln,business_date",
        "$first": str(RCE_PAGE_SIZE),
    }
    rows: list[dict[str, Any]] = []
    try:
        while url:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=60)) as response:
                if response.status != 200:
                    raise RceError(f"PSE answered {response.status}")
                data = await response.json()
            rows.extend(data.get("value", []))
            url, params = data.get("nextLink"), None
    except (aiohttp.ClientError, TimeoutError, ValueError) as err:
        raise RceError(str(err)) from err
    return rows


class RceCache:
    """The hourly RCE of the days asked for, fetched once and kept in a Store."""

    def __init__(self, store: _Store, fetch: RceFetch) -> None:
        """Keep the prices in store and fetch the missing days with fetch(first, last)."""
        self._store = store
        self._fetch = fetch
        self._hours: dict[int, float] | None = None
        self._days: set[str] = set()

    async def async_prices(self, first: date, last: date, today: date) -> dict[int, float]:
        """Return every known hourly price, after fetching the days from first to last it lacks.

        A day counts as fetched once PSE returned prices for it, or once it is
        older than RCE_MISSING_DAY_GRACE_DAYS before today without any.  A
        failed fetch is logged and leaves its days to the next call; their
        hours stay unpriced meanwhile.
        """
        if self._hours is None:
            data = await self._store.async_load() or {}
            self._hours = {int(hour): price for hour, price in data.get("hours", {}).items()}
            self._days = set(data.get("days", []))

        changed = False
        for start, end in self._missing_ranges(first, last):
            try:
                rows = await self._fetch(start, end)
            except RceError as err:
                _LOGGER.warning("Could not fetch RCE prices for %s to %s: %s", start, end, err)
                continue
            self._hours.update(hourly_prices(rows))
            published = {row.get("business_date") for row in rows}
            day = start
            while day <= end:
                if day.isoformat() in published or (today - day).days > RCE_MISSING_DAY_GRACE_DAYS:
                    self._days.add(day.isoformat())
                day += timedelta(days=1)
            changed = True

        if changed:
            await self._store.async_save(
                {"hours": {str(hour): price for hour, price in self._hours.items()}, "days": sorted(self._days)}
            )
        return dict(self._hours)

    def _missing_ranges(self, first: date, last: date) -> list[tuple[date, date]]:
        """Return the runs of consecutive days from first to last not fetched yet."""
        ranges: list[tuple[date, date]] = []
        day = first
        while day <= last:
            if day.isoformat() not in self._days:
                if ranges and ranges[-1][1] == day - timedelta(days=1):
                    ranges[-1] = (ranges[-1][0], day)
                else:
                    ranges.append((day, day))
            day += timedelta(days=1)
        return ranges
