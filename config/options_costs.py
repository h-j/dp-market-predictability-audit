"""
Options Transaction Costs Configuration.

Single source of truth for transaction fees, taxes, and slippage in options strategy simulations.
"""

# Flat brokerage per order (₹20/order)
BROKERAGE_PER_ORDER: float = 20.0

# Securities Transaction Tax (STT) on sell side premium (0.0625%)
STT_SELL_PREMIUM_PCT: float = 0.000625

# Slippage per leg (0.5% of option premium)
SLIPPAGE_PER_LEG_PCT: float = 0.005

# Exchange turnover charges + GST approximation (0.06% of premium)
EXCHANGE_GST_PCT: float = 0.0006


def calculate_leg_cost(premium: float, is_sell: bool = False, lot_size: int = 50) -> float:
    """
    Calculate friction costs for a single option leg (1 order = 1 leg).
    Includes flat brokerage, STT (if sell order), exchange charges/GST, and slippage.
    """
    notional_premium = premium * lot_size
    brokerage = BROKERAGE_PER_ORDER
    stt = (notional_premium * STT_SELL_PREMIUM_PCT) if is_sell else 0.0
    exchange_gst = notional_premium * EXCHANGE_GST_PCT
    slippage = notional_premium * SLIPPAGE_PER_LEG_PCT

    total_cost = brokerage + stt + exchange_gst + slippage
    return float(round(total_cost, 2))
