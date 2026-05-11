// Package vector converts the transaction payload into a 14-dimensional
// int16-quantized vector ready for KNN search.
//
// Phase 1c (this file): stdlib encoding/json for parsing, in-line normalization
// + quantization. A fused, zero-allocation parser will replace the stdlib JSON
// step in a later phase (see docs/lectures/03-runtime-and-infra.md).
//
// Quantization scheme is shared with the dataset package: real values in [0,1]
// → [0, 32000]; the -1 sentinel → -32000.
package vector

import (
	"encoding/json"
	"errors"
	"time"

	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
)

// Normalization constants — verbatim from resources/normalization.json.
const (
	MaxAmount            = 10000.0
	MaxInstallments      = 12.0
	AmountVsAvgRatio     = 10.0
	MaxMinutes           = 1440.0
	MaxKm                = 1000.0
	MaxTxCount24h        = 20.0
	MaxMerchantAvgAmount = 10000.0
)

// MccRisk maps merchant-category codes to risk values (dim 12). Codes not in
// the table use DefaultMccRisk. Verbatim from resources/mcc_risk.json.
var MccRisk = map[string]float32{
	"5411": 0.15,
	"5812": 0.30,
	"5912": 0.20,
	"5944": 0.45,
	"7801": 0.80,
	"7802": 0.75,
	"7995": 0.85,
	"4511": 0.35,
	"5311": 0.25,
	"5999": 0.50,
}

const DefaultMccRisk float32 = 0.5

// payload is the on-the-wire shape of POST /fraud-score.
type payload struct {
	ID          string `json:"id"`
	Transaction struct {
		Amount       float64 `json:"amount"`
		Installments int     `json:"installments"`
		RequestedAt  string  `json:"requested_at"`
	} `json:"transaction"`
	Customer struct {
		AvgAmount      float64  `json:"avg_amount"`
		TxCount24h     int      `json:"tx_count_24h"`
		KnownMerchants []string `json:"known_merchants"`
	} `json:"customer"`
	Merchant struct {
		ID        string  `json:"id"`
		MCC       string  `json:"mcc"`
		AvgAmount float64 `json:"avg_amount"`
	} `json:"merchant"`
	Terminal struct {
		IsOnline    bool    `json:"is_online"`
		CardPresent bool    `json:"card_present"`
		KmFromHome  float64 `json:"km_from_home"`
	} `json:"terminal"`
	LastTransaction *struct {
		Timestamp     string  `json:"timestamp"`
		KmFromCurrent float64 `json:"km_from_current"`
	} `json:"last_transaction"`
}

// Vectorize parses a JSON-encoded POST /fraud-score body and writes the 14
// int16-quantized dimensions into out.
func Vectorize(body []byte, out *[14]int16) error {
	var p payload
	if err := json.Unmarshal(body, &p); err != nil {
		return err
	}

	tx := p.Transaction
	cust := p.Customer
	merch := p.Merchant
	term := p.Terminal

	reqTime, err := time.Parse(time.RFC3339, tx.RequestedAt)
	if err != nil {
		return errors.New("invalid transaction.requested_at")
	}
	reqUTC := reqTime.UTC()
	reqHour := reqUTC.Hour()
	reqMin := reqUTC.Minute()
	// Day-of-week formula: data generator uses (days_since_epoch % 7), where
	// the QRust reference's isoUtcDayNumber sets January 1 of year 1 as day 0.
	// We compute the same with Go's Date+UTC and offset to make the encoding
	// match: mon=0..sun=6 ↔ (Weekday+6)%7. Empirically both match (Wed→2 etc.).
	reqDayOfWeek := (int(reqUTC.Weekday()) + 6) % 7
	// reqMinutes = day_number_since_epoch * 1440 + hour*60 + minute, integer.
	// Used only as `reqMinutes - lastMinutes` for dim 5.
	reqDayNum := dayNumberSinceYear1(reqUTC)
	reqMinutes := reqDayNum*1440 + reqHour*60 + reqMin

	out[0] = dataset.Quantize(float32(tx.Amount / MaxAmount))
	out[1] = dataset.Quantize(float32(float64(tx.Installments) / MaxInstallments))
	if cust.AvgAmount > 0 {
		out[2] = dataset.Quantize(float32((tx.Amount / cust.AvgAmount) / AmountVsAvgRatio))
	} else {
		out[2] = dataset.QuantScale
	}
	out[3] = dataset.Quantize(float32(reqHour) / 23.0)
	out[4] = dataset.Quantize(float32(reqDayOfWeek) / 6.0)

	if p.LastTransaction != nil {
		lastTime, err := time.Parse(time.RFC3339, p.LastTransaction.Timestamp)
		if err != nil {
			return errors.New("invalid last_transaction.timestamp")
		}
		lastUTC := lastTime.UTC()
		lastMinutes := dayNumberSinceYear1(lastUTC)*1440 + lastUTC.Hour()*60 + lastUTC.Minute()
		out[5] = dataset.Quantize(float32(float64(reqMinutes-lastMinutes) / MaxMinutes))
		out[6] = dataset.Quantize(float32(p.LastTransaction.KmFromCurrent / MaxKm))
	} else {
		out[5] = dataset.SentinelInt
		out[6] = dataset.SentinelInt
	}

	out[7] = dataset.Quantize(float32(term.KmFromHome / MaxKm))
	out[8] = dataset.Quantize(float32(float64(cust.TxCount24h) / MaxTxCount24h))
	out[9] = boolQuant(term.IsOnline)
	out[10] = boolQuant(term.CardPresent)
	out[11] = unknownMerchantQuant(merch.ID, cust.KnownMerchants)

	if risk, ok := MccRisk[merch.MCC]; ok {
		out[12] = dataset.Quantize(risk)
	} else {
		out[12] = dataset.Quantize(DefaultMccRisk)
	}

	out[13] = dataset.Quantize(float32(merch.AvgAmount / MaxMerchantAvgAmount))
	return nil
}

func boolQuant(b bool) int16 {
	if b {
		return dataset.QuantScale
	}
	return 0
}

var monthDaysBefore = [12]int{0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334}

// dayNumberSinceYear1 returns the day index where January 1 of year 1 is day 0.
// Matches the data generator's epoch (per the QRust reference implementation:
// references/qrust-luanmonteiro/src/fast-json.ts isoUtcDayNumber).
func dayNumberSinceYear1(t time.Time) int {
	y, mo, d := t.Date()
	yPrev := y - 1
	days := yPrev*365 + yPrev/4 - yPrev/100 + yPrev/400
	days += monthDaysBefore[int(mo)-1] + d - 1
	if mo > time.February && isLeap(y) {
		days++
	}
	return days
}

func isLeap(y int) bool {
	return (y%4 == 0 && y%100 != 0) || y%400 == 0
}

// unknownMerchantQuant returns QuantScale if merchantID is not in
// knownMerchants, 0 otherwise. Polarity is inverted vs the natural reading:
// 1 = unknown (riskier).
func unknownMerchantQuant(merchantID string, knownMerchants []string) int16 {
	for _, m := range knownMerchants {
		if m == merchantID {
			return 0
		}
	}
	return dataset.QuantScale
}
