"""
Regime-Filtered Options Strategy Simulator.

Executes weekly options trading policies based on Model C volatility forecasts vs India VIX.
Evaluates unconditional variance risk premium harvesting vs model-filtered policies across
naked (short strangle) and defined-risk (iron condor) structures.
"""

from dataclasses import dataclass
from typing import List, Optional

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
    vol_forecast: float  # Annualized percentage (vol_forecast_ann)
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
        # Unit check assertion: median vol_forecast_ann must be within 0.3x - 3x of median vix_close
        if "vol_forecast_5d" in df_weekly.columns and df_weekly["vol_forecast_5d"].notna().any():
            valid_fc = df_weekly["vol_forecast_5d"].dropna()
            med_forecast_ann = float(np.median(valid_fc * np.sqrt(252)))
            med_vix = float(np.median(df_weekly["vix_close"].dropna()))
            assert 0.3 * med_vix <= med_forecast_ann <= 3.0 * med_vix, (
                f"Unit error assertion failed: median vol_forecast_ann ({med_forecast_ann:.2f}) "
                f"is outside 0.3x-3x band of median vix_close ({med_vix:.2f})"
            )

        logs: List[WeeklyStrategyLog] = []

        for i in range(len(df_weekly) - 1):
            row_entry = df_weekly.iloc[i]
            row_expiry = df_weekly.iloc[i + 1]

            entry_date = str(row_entry["date"])
            expiry_date = str(row_expiry["date"])
            S_entry = float(row_entry["close"])
            S_expiry = float(row_expiry["close"])
            vix_entry = float(row_entry["vix_close"])

            # Explicit annualization without heuristic unit guessing
            vol_forecast_daily = float(row_entry.get("vol_forecast_5d", vix_entry / np.sqrt(252)))
            vol_forecast_ann = float(vol_forecast_daily * np.sqrt(252))

            ratio = vol_forecast_ann / vix_entry if vix_entry > 0 else 1.0

            action = "NO_TRADE"
            trade_res: Optional[StructureTradeResult] = None
            pnl = 0.0

            if policy_name == "always_sell":
                action = f"SELL_{self.structure.upper()}"
                trade_res = self.pricer.simulate_structure(self.structure, S_entry, S_expiry, vix_entry)
                pnl = trade_res.net_pnl

            elif policy_name == "sell_when_calm":
                if vol_forecast_ann < (vix_entry * calm_k):
                    action = f"SELL_{self.structure.upper()}"
                    trade_res = self.pricer.simulate_structure(self.structure, S_entry, S_expiry, vix_entry)
                    pnl = trade_res.net_pnl

            elif policy_name == "buy_when_storm":
                if vol_forecast_ann > (vix_entry * 1.2) or vix_entry > 20.0:
                    action = "BUY_LONG_STRADDLE"
                    trade_res = self.pricer.simulate_structure("long_straddle", S_entry, S_expiry, vix_entry)
                    pnl = trade_res.net_pnl

            elif policy_name == "combined_regime":
                if vol_forecast_ann < (vix_entry * calm_k):
                    action = f"SELL_{self.structure.upper()}"
                    trade_res = self.pricer.simulate_structure(self.structure, S_entry, S_expiry, vix_entry)
                    pnl = trade_res.net_pnl
                elif vol_forecast_ann > (vix_entry * 1.2) or vix_entry > 20.0:
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
                    vol_forecast=round(vol_forecast_ann, 4),
                    ratio_forecast_vix=round(ratio, 4),
                    policy_name=policy_name,
                    action_taken=action,
                    trade_result=trade_res,
                    pnl_rupees=round(pnl, 2),
                )
            )

        return logs
