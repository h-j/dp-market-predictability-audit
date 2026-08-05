"""
Regime-Filtered Options Strategy Simulator.

Executes weekly options trading policies based on Model C volatility forecasts vs India VIX.
Evaluates unconditional variance risk premium harvesting vs model-filtered policies across
naked (short strangle) and defined-risk (iron condor) structures.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from market.options.synthetic_pricer import StructureTradeResult, SyntheticOptionPricer


@dataclass
class WeeklyStrategyLog:
    week_index: int
    entry_date: str
    expiry_date: str
    entry_spot: float
    expiry_spot: float
    vix_entry: float
    vol_forecast: float
    ratio_forecast_vix: float
    policy_name: str
    action_taken: str
    trade_result: Optional[StructureTradeResult]
    pnl_rupees: float


class OptionsRegimeStrategyEngine:
    """
    Executes weekly regime-filtered options trading simulation over historical walk-forward predictions.
    """

    def __init__(
        self,
        pricer: Optional[SyntheticOptionPricer] = None,
        starting_capital: float = 1000000.0,  # ₹10 Lakhs capital base
        structure: str = "short_strangle",
    ):
        self.pricer = pricer if pricer is not None else SyntheticOptionPricer()
        self.starting_capital = starting_capital
        self.structure = structure

    def run_simulation(
        self,
        df_weekly: pd.DataFrame,
        policy_name: str = "always_sell",
        calm_k: float = 1.0,
    ) -> List[WeeklyStrategyLog]:
        """
        df_weekly columns required: date, close, vix_close, vol_forecast_5d
        Cadence: weekly rows (stride 5)
        """
        logs: List[WeeklyStrategyLog] = []

        for i in range(len(df_weekly) - 1):
            row_entry = df_weekly.iloc[i]
            row_expiry = df_weekly.iloc[i + 1]

            entry_date = str(row_entry["date"])
            expiry_date = str(row_expiry["date"])
            S_entry = float(row_entry["close"])
            S_expiry = float(row_expiry["close"])
            vix_entry = float(row_entry["vix_close"])
            vol_forecast = float(row_entry.get("vol_forecast_5d", vix_entry / 100.0 * np.sqrt(252)))

            # Convert forecast to annual percentage if in decimal format
            if vol_forecast < 1.0:
                vol_forecast_pct = vol_forecast * 100.0
            else:
                vol_forecast_pct = vol_forecast

            ratio = vol_forecast_pct / vix_entry if vix_entry > 0 else 1.0

            action = "NO_TRADE"
            trade_res: Optional[StructureTradeResult] = None
            pnl = 0.0

            if policy_name == "always_sell":
                action = f"SELL_{self.structure.upper()}"
                trade_res = self.pricer.simulate_structure(self.structure, S_entry, S_expiry, vix_entry)
                pnl = trade_res.net_pnl

            elif policy_name == "sell_when_calm":
                if vol_forecast_pct < (vix_entry * calm_k):
                    action = f"SELL_{self.structure.upper()}"
                    trade_res = self.pricer.simulate_structure(self.structure, S_entry, S_expiry, vix_entry)
                    pnl = trade_res.net_pnl

            elif policy_name == "buy_when_storm":
                if vol_forecast_pct > (vix_entry * 1.2):
                    action = "BUY_LONG_STRADDLE"
                    trade_res = self.pricer.simulate_structure("long_straddle", S_entry, S_expiry, vix_entry)
                    pnl = trade_res.net_pnl

            elif policy_name == "combined_regime":
                if vol_forecast_pct < (vix_entry * calm_k):
                    action = f"SELL_{self.structure.upper()}"
                    trade_res = self.pricer.simulate_structure(self.structure, S_entry, S_expiry, vix_entry)
                    pnl = trade_res.net_pnl
                elif vol_forecast_pct > (vix_entry * 1.2):
                    action = "BUY_LONG_STRADDLE"
                    trade_res = self.pricer.simulate_structure("long_straddle", S_entry, S_expiry, vix_entry)
                    pnl = trade_res.net_pnl

            logs.append(
                WeeklyStrategyLog(
                    week_index=i + 1,
                    entry_date=entry_date,
                    expiry_date=expiry_date,
                    entry_spot=S_entry,
                    expiry_spot=S_expiry,
                    vix_entry=vix_entry,
                    vol_forecast=vol_forecast_pct,
                    ratio_forecast_vix=round(ratio, 4),
                    policy_name=policy_name,
                    action_taken=action,
                    trade_result=trade_res,
                    pnl_rupees=round(pnl, 2),
                )
            )

        return logs
