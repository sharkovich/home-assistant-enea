"""A prosumer's net-billing: returned energy valued at the hourly RCE × 1.23."""
from __future__ import annotations

import datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from custom_components.enea import config_flow, costs, rce
from custom_components.enea.const import STAT_KEY_ENERGY_RETURNED

UTC = datetime.UTC
DAY = datetime.date(2026, 6, 15)


def _row(end_utc: str, period_utc: str, price_mwh: float) -> dict[str, Any]:
    """One row of the PSE rce-pln report."""
    return {"dtime_utc": end_utc, "period_utc": period_utc, "rce_pln": price_mwh}


def _hour(hour: int) -> int:
    """The UTC timestamp of the start of an hour on DAY."""
    return int(datetime.datetime(2026, 6, 15, hour, tzinfo=UTC).timestamp())


def test_quarter_hour_prices_average_into_the_hour_they_start_in() -> None:
    rows = [
        _row("2026-06-15 10:15:00", "10:00 - 10:15", 400.0),
        _row("2026-06-15 10:30:00", "10:15 - 10:30", 300.0),
        _row("2026-06-15 10:45:00", "10:30 - 10:45", 200.0),
        _row("2026-06-15 11:00:00", "10:45 - 11:00", 100.0),
        _row("2026-06-16 00:00:00", "23:45 - 00:00", -50.0),
    ]

    prices = rce.hourly_prices(rows)

    # PLN/MWh → PLN/kWh; the quarter ending at midnight belongs to 23:00.
    assert prices == {_hour(10): pytest.approx(0.25), _hour(23): pytest.approx(-0.05)}


def test_hourly_prices_keep_their_own_hour() -> None:
    """Before PSE moved to 15-minute periods a row covered a whole hour."""
    prices = rce.hourly_prices([_row("2026-06-15 13:00:00", "12:00 - 13:00", 480.0)])

    assert prices == {_hour(12): pytest.approx(0.48)}


class _MemoryStore:
    """A Store stand-in that keeps the data in memory."""

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self.data = data

    async def async_load(self) -> dict[str, Any] | None:
        return self.data

    async def async_save(self, data: dict[str, Any]) -> None:
        self.data = data


def _day_rows(day: datetime.date, price_mwh: float = 400.0) -> list[dict[str, Any]]:
    """24 hourly rows for one day (UTC hours, enough for the cache's bookkeeping)."""
    rows = []
    for hour in range(24):
        end = datetime.datetime(day.year, day.month, day.day, hour, tzinfo=UTC) + datetime.timedelta(hours=1)
        rows.append(
            {**_row(end.strftime("%Y-%m-%d %H:%M:%S"), f"{hour:02d}:00 - {(hour + 1) % 24:02d}:00", price_mwh),
             "business_date": day.isoformat()}
        )
    return rows


async def test_the_cache_fetches_only_days_it_does_not_have() -> None:
    calls: list[tuple[datetime.date, datetime.date]] = []

    async def fetch(first: datetime.date, last: datetime.date) -> list[dict[str, Any]]:
        calls.append((first, last))
        out: list[dict[str, Any]] = []
        day = first
        while day <= last:
            out += _day_rows(day)
            day += datetime.timedelta(days=1)
        return out

    store = _MemoryStore()
    cache = rce.RceCache(store, fetch)
    today = datetime.date(2026, 6, 20)

    first = await cache.async_prices(datetime.date(2026, 6, 14), datetime.date(2026, 6, 15), today)
    second = await cache.async_prices(datetime.date(2026, 6, 14), datetime.date(2026, 6, 16), today)

    assert calls == [
        (datetime.date(2026, 6, 14), datetime.date(2026, 6, 15)),
        (datetime.date(2026, 6, 16), datetime.date(2026, 6, 16)),
    ]
    assert first[_hour(10)] == pytest.approx(0.4)
    assert len(second) == 72
    # A new cache over the same store needs no fetch at all.
    again: list[Any] = []

    async def no_fetch(first: datetime.date, last: datetime.date) -> list[dict[str, Any]]:
        again.append((first, last))
        return []

    reloaded = await rce.RceCache(store, no_fetch).async_prices(
        datetime.date(2026, 6, 14), datetime.date(2026, 6, 16), today
    )
    assert again == []
    assert reloaded == second


async def test_a_recent_day_without_prices_is_asked_for_again() -> None:
    """PSE may not have published a day yet; only an old empty day is given up on."""
    calls: list[tuple[datetime.date, datetime.date]] = []

    async def fetch(first: datetime.date, last: datetime.date) -> list[dict[str, Any]]:
        calls.append((first, last))
        return []

    cache = rce.RceCache(_MemoryStore(), fetch)
    recent = datetime.date(2026, 6, 19)
    old = datetime.date(2026, 1, 1)
    today = datetime.date(2026, 6, 20)

    await cache.async_prices(old, old, today)
    await cache.async_prices(old, old, today)
    await cache.async_prices(recent, recent, today)
    await cache.async_prices(recent, recent, today)

    assert calls == [(old, old), (recent, recent), (recent, recent)]


async def test_a_failed_fetch_leaves_the_hours_unpriced() -> None:
    async def fetch(first: datetime.date, last: datetime.date) -> list[dict[str, Any]]:
        raise rce.RceError("boom")

    prices = await rce.RceCache(_MemoryStore(), fetch).async_prices(DAY, DAY, DAY + datetime.timedelta(days=5))

    assert prices == {}


class _CostPricing:
    energy = 0.5
    total_distribution = 0.3
    oze = 0.007
    cogeneration = 0.003


class _CostPeriod:
    def __init__(self) -> None:
        self.zones = {"peak": _CostPricing()}

    def get_zone_at_hour(self, hour: int, day: datetime.date | None = None) -> str:
        return "peak"


class _CostTariff:
    name = "G12w"

    def __init__(self) -> None:
        self.periods = [SimpleNamespace(valid_from=datetime.date(2026, 1, 1))]

    def get_period_for_date(self, d: datetime.date) -> _CostPeriod:
        return _CostPeriod()


def _returned(hour: int, kwh: float) -> list[tuple[datetime.date, dict[str, Any]]]:
    """One day with one hour of returned energy (the slot ending an hour later)."""
    end = datetime.datetime(2026, 6, 15, hour + 1, tzinfo=UTC)
    return [
        (DAY, {STAT_KEY_ENERGY_RETURNED: {
            "values": [{"integrationEnd": end.timestamp() * 1000, "items": [{"value": kwh}]}]
        }})
    ]


async def _inject(monkeypatch: pytest.MonkeyPatch, days: list, prices: dict[int, float]) -> list:
    """Run the cost injection for returned energy and return what it would write."""
    injected: list[tuple[str, list]] = []

    async def fake_inject(hass: Any, meter_code: str, name: str, series: list) -> float:
        injected.append((name, series))
        return 0.0

    monkeypatch.setattr(costs, "_inject_cost_series", fake_inject)
    await costs.async_insert_cost_statistics(
        object(), "PPE", days, _CostTariff(), False, True, returned_prices=prices
    )
    return injected


async def test_returned_energy_is_valued_at_the_hours_rce(monkeypatch: pytest.MonkeyPatch) -> None:
    injected = await _inject(monkeypatch, _returned(10, 2.0), {_hour(10): 0.25})

    # 2 kWh × 0.25 PLN/kWh × 1.23 = 0.615 zł — the tariff's price plays no part.
    assert injected[0][0] == "Koszt energii oddana – Szczyt"
    assert injected[0][1][0][1] == pytest.approx(0.615)


async def test_a_negative_rce_values_returned_energy_at_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    days = _returned(10, 2.0)
    days[0][1][STAT_KEY_ENERGY_RETURNED]["values"] += _returned(11, 3.0)[0][1][STAT_KEY_ENERGY_RETURNED]["values"]

    injected = await _inject(monkeypatch, days, {_hour(10): -0.05, _hour(11): 0.25})

    # The negative hour adds nothing to the deposit (it is not charged for either).
    assert [cost for _, cost in injected[0][1]] == [0.0, pytest.approx(3.0 * 0.25 * 1.23)]


async def test_an_hour_without_rce_is_left_unpriced(monkeypatch: pytest.MonkeyPatch) -> None:
    injected = await _inject(monkeypatch, _returned(10, 2.0), {})

    assert injected == []


def test_the_days_rce_is_part_of_its_fingerprint() -> None:
    """Prices published later, or corrected, must reprice the day."""
    tariff = _CostTariff()

    missing = costs.price_signatures(tariff, DAY, DAY, returned_prices={})
    known = costs.price_signatures(tariff, DAY, DAY, returned_prices={_hour(10): 0.25})
    other = costs.price_signatures(tariff, DAY, DAY, returned_prices={_hour(10): 0.30})

    assert len({missing[DAY.isoformat()], known[DAY.isoformat()], other[DAY.isoformat()]}) == 3
    assert missing != costs.price_signatures(tariff, DAY, DAY)


def _options_flow(options: dict[str, Any]) -> config_flow.EneaOptionsFlow:
    """An options flow over a prosumer's loaded entry."""
    entry = SimpleNamespace(
        options=options,
        runtime_data=SimpleNamespace(coordinator=SimpleNamespace(prosumer=True)),
    )
    flow = config_flow.EneaConfigFlow.async_get_options_flow(entry)
    flow.handler = "entry"
    flow.flow_id = "options-flow"
    flow.hass = SimpleNamespace(
        config_entries=SimpleNamespace(async_get_known_entry=Mock(return_value=entry))
    )
    return flow


async def test_a_prosumer_can_choose_net_billing() -> None:
    options = {
        "update_interval": {"hours": 3, "minutes": 30},
        "fetch_consumption": True,
        "fetch_generation": True,
        "fetch_power_consumption": False,
        "fetch_power_generation": False,
        "net_metering": "net_billing",
    }

    result = await _options_flow(options).async_step_init()

    assert result["data_schema"]({})["net_metering"] == "net_billing"


def _coordinator(*, prosumer: bool, net_billing: bool, prices: dict[int, float]) -> Any:
    """A coordinator with only what pricing returned energy needs."""
    from custom_components.enea.coordinator import EneaUpdateCoordinator

    async def fetch(first: datetime.date, last: datetime.date) -> list[dict[str, Any]]:
        return []

    coord = object.__new__(EneaUpdateCoordinator)
    coord._prosumer = prosumer
    coord._net_metering_ratio = None
    coord._net_billing = net_billing
    stored = {"hours": {str(hour): price for hour, price in prices.items()}, "days": [DAY.isoformat()]}
    coord._rce = rce.RceCache(_MemoryStore(stored), fetch)
    coord._cost_prices_store = _MemoryStore()
    coord._cost_prices = None
    coord._cost_prices_loaded = False
    return coord


@pytest.mark.parametrize(
    ("prosumer", "net_billing", "priced"),
    [
        pytest.param(True, True, True, id="net-billing-prosumer"),
        pytest.param(True, False, False, id="net-metering-or-none"),
        pytest.param(False, True, False, id="not-a-prosumer"),
    ],
)
async def test_only_a_net_billing_prosumer_values_returned_energy_at_rce(prosumer, net_billing, priced) -> None:
    coord = _coordinator(prosumer=prosumer, net_billing=net_billing, prices={_hour(10): 0.25})

    prices = await coord._async_returned_prices(DAY, DAY)

    assert (prices == {_hour(10): 0.25}) if priced else prices is None
    assert coord._returned_cost_ratio == 1.0


async def test_the_stored_fingerprints_include_the_rce() -> None:
    """Saved fingerprints must match what the next repricing check computes."""
    coord = _coordinator(prosumer=True, net_billing=True, prices={_hour(10): 0.25})
    tariff = _CostTariff()

    await coord._async_save_cost_prices(tariff, DAY, DAY)

    expected = costs.price_signatures(tariff, DAY, DAY, returned_prices={_hour(10): 0.25})
    assert coord._cost_prices_store.data == {"days": expected}
