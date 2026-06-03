#!/usr/bin/env python3
"""
Daily S&P 500 (SPY) and NASDAQ 100 (QQQ) market analysis
Sends Telegram notification after NYSE market close
"""

import os
import sys
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# ─── Data Fetching ────────────────────────────────────────────────────────────

def fetch(ticker: str, period: str = "1y") -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    return df


# ─── Indicators ───────────────────────────────────────────────────────────────

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=True).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=True).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger(series: pd.Series, period=20, n_std=2):
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return ma, ma + n_std * std, ma - n_std * std


# ─── Analysis ─────────────────────────────────────────────────────────────────

def analyze(ticker: str, display_name: str) -> dict:
    df = fetch(ticker)
    close = df["Close"]
    volume = df["Volume"]

    # Check data freshness — warn if latest bar is not from today or yesterday
    et = timezone(timedelta(hours=-4))  # EDT; close enough year-round for a warning
    latest_date = close.index[-1].date()
    today = datetime.now(et).date()
    is_stale = (today - latest_date).days > 1

    price = float(close.iloc[-1])
    prev = float(close.iloc[-2])
    daily_pct = (price - prev) / prev * 100

    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1])
    ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

    rsi_val = float(rsi(close).iloc[-1])

    macd_line, sig_line, hist = macd(close)
    macd_now = float(macd_line.iloc[-1])
    sig_now = float(sig_line.iloc[-1])
    hist_now = float(hist.iloc[-1])
    hist_prev = float(hist.iloc[-2])

    _, bb_upper, bb_lower = bollinger(close)
    bb_pct = (price - float(bb_lower.iloc[-1])) / (
        float(bb_upper.iloc[-1]) - float(bb_lower.iloc[-1])
    ) * 100

    avg_vol = float(volume.rolling(20).mean().iloc[-1])
    vol_ratio = float(volume.iloc[-1]) / avg_vol

    # ── Scoring ──────────────────────────────────────────────────────────────
    score = 0.0
    signals = []

    # Trend: price vs moving averages
    if price > ma20:
        score += 1.0
        signals.append("✅ 价格站上 MA20")
    else:
        score -= 1.0
        signals.append("❌ 价格跌破 MA20")

    if price > ma50:
        score += 1.5
        signals.append("✅ 价格站上 MA50")
    else:
        score -= 1.5
        signals.append("❌ 价格跌破 MA50")

    if ma200 is not None:
        if price > ma200:
            score += 2.0
            signals.append("✅ 价格站上 MA200（牛市结构）")
        else:
            score -= 2.0
            signals.append("❌ 价格跌破 MA200（熊市警示）")

    if ma20 > ma50:
        score += 0.5
        signals.append("✅ MA20 > MA50（短期上行）")
    else:
        score -= 0.5
        signals.append("⚠️ MA20 < MA50（短期走弱）")

    # Momentum: RSI
    if rsi_val < 30:
        score += 2.0
        signals.append(f"🔥 RSI {rsi_val:.1f} 超卖，反弹机会")
    elif rsi_val < 45:
        score += 1.0
        signals.append(f"📊 RSI {rsi_val:.1f} 偏弱未超卖")
    elif rsi_val < 55:
        signals.append(f"📊 RSI {rsi_val:.1f} 中性")
    elif rsi_val < 70:
        score -= 0.5
        signals.append(f"📊 RSI {rsi_val:.1f} 偏强")
    else:
        score -= 2.0
        signals.append(f"⚠️ RSI {rsi_val:.1f} 超买，注意回调")

    # Momentum: MACD
    if macd_now > sig_now and hist_now > hist_prev:
        score += 2.0
        signals.append("✅ MACD 金叉 / 动能增强")
    elif macd_now > sig_now:
        score += 0.5
        signals.append("📊 MACD 多头，动能减弱中")
    elif macd_now < sig_now and hist_now < hist_prev:
        score -= 2.0
        signals.append("❌ MACD 死叉 / 动能减弱")
    else:
        score -= 0.5
        signals.append("📊 MACD 空头，动能改善中")

    # Bollinger Band position
    if bb_pct < 20:
        score += 1.0
        signals.append(f"🔥 布林带下轨附近 ({bb_pct:.0f}%)，超卖")
    elif bb_pct > 80:
        score -= 1.0
        signals.append(f"⚠️ 布林带上轨附近 ({bb_pct:.0f}%)，超买")
    else:
        signals.append(f"📊 布林带中性 ({bb_pct:.0f}%)")

    # Volume
    if daily_pct > 0 and vol_ratio > 1.2:
        score += 0.5
        signals.append(f"✅ 上涨放量 ({vol_ratio:.1f}x 均量)")
    elif daily_pct < 0 and vol_ratio > 1.2:
        score -= 0.5
        signals.append(f"⚠️ 下跌放量 ({vol_ratio:.1f}x 均量)，抛压较重")
    elif vol_ratio < 0.7:
        signals.append(f"📊 缩量 ({vol_ratio:.1f}x 均量)")
    else:
        signals.append(f"📊 成交量正常 ({vol_ratio:.1f}x 均量)")

    # ── Overall recommendation ────────────────────────────────────────────────
    if score >= 4.0:
        rec = "🟢 买入"
        detail = "多项指标共振向上，可考虑买入或加仓"
    elif score >= 1.5:
        rec = "🔵 偏多观望"
        detail = "趋势偏多，可小仓布局或等待更好入场点"
    elif score >= -1.5:
        rec = "⚪ 观望"
        detail = "多空信号混杂，建议观望等待明确方向"
    elif score >= -4.0:
        rec = "🟡 偏空观望"
        detail = "趋势偏弱，不宜买入，持仓者注意止损"
    else:
        rec = "🔴 回避"
        detail = "空头信号明显，建议回避或减仓"

    return {
        "name": display_name,
        "ticker": ticker,
        "date": latest_date.strftime("%Y-%m-%d"),
        "is_stale": is_stale,
        "price": price,
        "daily_pct": daily_pct,
        "ma20": ma20,
        "ma50": ma50,
        "ma200": ma200,
        "rsi": rsi_val,
        "bb_pct": bb_pct,
        "vol_ratio": vol_ratio,
        "score": score,
        "rec": rec,
        "detail": detail,
        "signals": signals,
    }


# ─── Message Formatting ───────────────────────────────────────────────────────

def _ticker_block(d: dict) -> str:
    arrow = "📈" if d["daily_pct"] >= 0 else "📉"
    sign = "+" if d["daily_pct"] >= 0 else ""
    stale_note = "  ⚠️ 数据非最新交易日" if d["is_stale"] else ""

    ma_line = f"MA20: {d['ma20']:.2f}  MA50: {d['ma50']:.2f}"
    if d["ma200"] is not None:
        ma_line += f"  MA200: {d['ma200']:.2f}"

    signal_lines = "\n".join(f"  {s}" for s in d["signals"])

    return (
        f"*{d['name']} ({d['ticker']})*{stale_note}\n"
        f"收盘: ${d['price']:.2f}  {arrow} {sign}{d['daily_pct']:.2f}%\n"
        f"\n"
        f"均线:  {ma_line}\n"
        f"RSI:   {d['rsi']:.1f}   布林带位置: {d['bb_pct']:.0f}%   量比: {d['vol_ratio']:.1f}x\n"
        f"\n"
        f"信号分析:\n"
        f"{signal_lines}\n"
        f"\n"
        f"综合得分: {d['score']:.1f}\n"
        f"📌 操作建议: {d['rec']}\n"
        f"_{d['detail']}_"
    )


def build_message(spy: dict, qqq: dict) -> str:
    now_et = datetime.now(timezone(timedelta(hours=-4)))
    date_str = now_et.strftime("%Y-%m-%d")
    divider = "─" * 28

    return (
        f"📊 *每日美股分析 — {date_str}*\n"
        f"{divider}\n\n"
        f"{_ticker_block(spy)}\n\n"
        f"{divider}\n\n"
        f"{_ticker_block(qqq)}\n\n"
        f"{divider}\n"
        f"_数据: Yahoo Finance  |  仅供参考，非投资建议_"
    )


# ─── Telegram ────────────────────────────────────────────────────────────────

def send_telegram(token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(
        url,
        json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
        timeout=30,
    )
    resp.raise_for_status()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print("ERROR: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set", file=sys.stderr)
        sys.exit(1)

    print("Fetching SPY...")
    spy = analyze("SPY", "标普500 ETF")
    print(f"  {spy['date']}  {spy['daily_pct']:+.2f}%  score={spy['score']:.1f}  {spy['rec']}")

    print("Fetching QQQ...")
    qqq = analyze("QQQ", "纳斯达克100 ETF")
    print(f"  {qqq['date']}  {qqq['daily_pct']:+.2f}%  score={qqq['score']:.1f}  {qqq['rec']}")

    msg = build_message(spy, qqq)
    print("\n--- Message preview ---")
    print(msg)
    print("-----------------------\n")

    send_telegram(token, chat_id, msg)
    print("Telegram message sent.")


if __name__ == "__main__":
    main()
