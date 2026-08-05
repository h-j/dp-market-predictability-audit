"""
Synthetic Option Pricer & Multi-Leg Structure Simulator.

Provides European Black-Scholes pricing using spot price S, strike K, time-to-expiry T,
and India VIX as ATM implied volatility input. Calculates multi-leg payoffs and net P&L
after friction costs.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import norm

from config.options_costs import calculate_leg_cost


def bs_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """
    Black-Scholes European Option Pricing.
    S: Spot price
    K: Strike price
    T: Time to expiry in years
    r: Risk-free rate (annualized decimal, e.g. 0.07)
    sigma: Implied volatility (annualized decimal, e.g. 0.15 for VIX=15)
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, S - K) if option_type.lower() == "call" else max(0.0, K - S)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return float(max(0.0, price))


@dataclass
class StructureTradeResult:
    structure_name: str
    entry_spot: float
    expiry_spot: float
    vix_entry: float
    gross_credit_debit: float  # Positive = credit received, Negative = debit paid
    friction_cost: float       # Total brokerage, STT, exchange, slippage
    expiry_payoff: float       # Payoff at expiry from spot move
    net_pnl: float             # Net P&L in currency units (₹)


class SyntheticOptionPricer:
    """
    Simulates multi-leg option structures on NIFTY index using Black-Scholes pricing and India VIX.
    """

    def __init__(self, risk_free_rate: float = 0.07, lot_size: int = 50, expiry_days: float = 7.0):
        self.r = risk_free_rate
        self.lot_size = lot_size
        self.T = expiry_days / 365.0

    def simulate_structure(
        self, structure_name: str, S_entry: float, S_expiry: float, vix_entry: float
    ) -> StructureTradeResult:
        sigma = vix_entry / 100.0
        struct = structure_name.lower()

        if struct == "short_straddle":
            # Sell ATM Call + Sell ATM Put
            call_prem = bs_price(S_entry, S_entry, self.T, self.r, sigma, "call")
            put_prem = bs_price(S_entry, S_entry, self.T, self.r, sigma, "put")

            credit_per_unit = call_prem + put_prem
            gross_credit = credit_per_unit * self.lot_size

            cost_c = calculate_leg_cost(call_prem, is_sell=True, lot_size=self.lot_size)
            cost_p = calculate_leg_cost(put_prem, is_sell=True, lot_size=self.lot_size)
            total_cost = cost_c + cost_p

            expiry_loss_unit = max(0.0, S_expiry - S_entry) + max(0.0, S_entry - S_expiry)
            expiry_payoff = gross_credit - (expiry_loss_unit * self.lot_size)
            net_pnl = expiry_payoff - total_cost

            return StructureTradeResult("short_straddle", S_entry, S_expiry, vix_entry, gross_credit, total_cost, expiry_payoff, net_pnl)

        elif struct == "short_strangle":
            # Sell +2% OTM Call & -2% OTM Put
            K_call = S_entry * 1.02
            K_put = S_entry * 0.98

            call_prem = bs_price(S_entry, K_call, self.T, self.r, sigma, "call")
            put_prem = bs_price(S_entry, K_put, self.T, self.r, sigma, "put")

            credit_per_unit = call_prem + put_prem
            gross_credit = credit_per_unit * self.lot_size

            cost_c = calculate_leg_cost(call_prem, is_sell=True, lot_size=self.lot_size)
            cost_p = calculate_leg_cost(put_prem, is_sell=True, lot_size=self.lot_size)
            total_cost = cost_c + cost_p

            expiry_loss_unit = max(0.0, S_expiry - K_call) + max(0.0, K_put - S_expiry)
            expiry_payoff = gross_credit - (expiry_loss_unit * self.lot_size)
            net_pnl = expiry_payoff - total_cost

            return StructureTradeResult("short_strangle", S_entry, S_expiry, vix_entry, gross_credit, total_cost, expiry_payoff, net_pnl)

        elif struct == "iron_condor":
            # Sell +2% Call, Long +4% Call; Sell -2% Put, Long -4% Put
            Kc_short = S_entry * 1.02
            Kc_long = S_entry * 1.04
            Kp_short = S_entry * 0.98
            Kp_long = S_entry * 0.96

            c_short = bs_price(S_entry, Kc_short, self.T, self.r, sigma, "call")
            c_long = bs_price(S_entry, Kc_long, self.T, self.r, sigma, "call")
            p_short = bs_price(S_entry, Kp_short, self.T, self.r, sigma, "put")
            p_long = bs_price(S_entry, Kp_long, self.T, self.r, sigma, "put")

            net_credit_unit = (c_short - c_long) + (p_short - p_long)
            gross_credit = net_credit_unit * self.lot_size

            cost_cs = calculate_leg_cost(c_short, is_sell=True, lot_size=self.lot_size)
            cost_cl = calculate_leg_cost(c_long, is_sell=False, lot_size=self.lot_size)
            cost_ps = calculate_leg_cost(p_short, is_sell=True, lot_size=self.lot_size)
            cost_pl = calculate_leg_cost(p_long, is_sell=False, lot_size=self.lot_size)
            total_cost = cost_cs + cost_cl + cost_ps + cost_pl

            call_payoff = max(0.0, S_expiry - Kc_short) - max(0.0, S_expiry - Kc_long)
            put_payoff = max(0.0, Kp_short - S_expiry) - max(0.0, Kp_long - S_expiry)
            total_loss_unit = call_payoff + put_payoff

            expiry_payoff = gross_credit - (total_loss_unit * self.lot_size)
            net_pnl = expiry_payoff - total_cost

            return StructureTradeResult("iron_condor", S_entry, S_expiry, vix_entry, gross_credit, total_cost, expiry_payoff, net_pnl)

        elif struct == "long_straddle":
            # Buy ATM Call + Buy ATM Put
            call_prem = bs_price(S_entry, S_entry, self.T, self.r, sigma, "call")
            put_prem = bs_price(S_entry, S_entry, self.T, self.r, sigma, "put")

            debit_per_unit = call_prem + put_prem
            gross_debit = -debit_per_unit * self.lot_size

            cost_c = calculate_leg_cost(call_prem, is_sell=False, lot_size=self.lot_size)
            cost_p = calculate_leg_cost(put_prem, is_sell=False, lot_size=self.lot_size)
            total_cost = cost_c + cost_p

            expiry_gain_unit = max(0.0, S_expiry - S_entry) + max(0.0, S_entry - S_expiry)
            expiry_payoff = (expiry_gain_unit * self.lot_size) + gross_debit
            net_pnl = expiry_payoff - total_cost

            return StructureTradeResult("long_straddle", S_entry, S_expiry, vix_entry, gross_debit, total_cost, expiry_payoff, net_pnl)

        else:
            raise ValueError(f"Unsupported structure {structure_name}")
