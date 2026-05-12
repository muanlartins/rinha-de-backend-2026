package vector

import (
	"encoding/json"
	"errors"
	"os"
	"testing"
	"time"

	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
)

// vectorizeSlow is a reference parser using encoding/json + time.Parse. Only
// compiled in tests, used as the oracle that VectorizeFast must match.
func vectorizeSlow(body []byte, out *[dataset.Stride]int16) error {
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
	var p payload
	if err := json.Unmarshal(body, &p); err != nil {
		return err
	}

	tx, cust, merch, term := p.Transaction, p.Customer, p.Merchant, p.Terminal
	reqTime, err := time.Parse(time.RFC3339, tx.RequestedAt)
	if err != nil {
		return errors.New("invalid transaction.requested_at")
	}
	reqUTC := reqTime.UTC()
	reqDayOfWeek := (int(reqUTC.Weekday()) + 6) % 7
	reqDayNum := dayNumSlow(reqUTC)
	reqMinutes := reqDayNum*1440 + reqUTC.Hour()*60 + reqUTC.Minute()

	out[0] = dataset.Quantize(float32(tx.Amount / MaxAmount))
	out[1] = dataset.Quantize(float32(float64(tx.Installments) / MaxInstallments))
	if cust.AvgAmount > 0 {
		out[2] = dataset.Quantize(float32((tx.Amount / cust.AvgAmount) / AmountVsAvgRatio))
	} else {
		out[2] = dataset.QuantScale
	}
	out[3] = dataset.Quantize(float32(reqUTC.Hour()) / 23.0)
	out[4] = dataset.Quantize(float32(reqDayOfWeek) / 6.0)

	if p.LastTransaction != nil {
		lastTime, err := time.Parse(time.RFC3339, p.LastTransaction.Timestamp)
		if err != nil {
			return errors.New("invalid last_transaction.timestamp")
		}
		lastUTC := lastTime.UTC()
		lastMinutes := dayNumSlow(lastUTC)*1440 + lastUTC.Hour()*60 + lastUTC.Minute()
		out[5] = dataset.Quantize(float32(float64(reqMinutes-lastMinutes) / MaxMinutes))
		out[6] = dataset.Quantize(float32(p.LastTransaction.KmFromCurrent / MaxKm))
	} else {
		out[5] = dataset.SentinelInt
		out[6] = dataset.SentinelInt
	}

	out[7] = dataset.Quantize(float32(term.KmFromHome / MaxKm))
	out[8] = dataset.Quantize(float32(float64(cust.TxCount24h) / MaxTxCount24h))
	if term.IsOnline {
		out[9] = dataset.QuantScale
	}
	if term.CardPresent {
		out[10] = dataset.QuantScale
	}
	out[11] = dataset.QuantScale
	for _, m := range cust.KnownMerchants {
		if m == merch.ID {
			out[11] = 0
			break
		}
	}
	if risk, ok := MccRisk[merch.MCC]; ok {
		out[12] = dataset.Quantize(risk)
	} else {
		out[12] = dataset.Quantize(DefaultMccRisk)
	}
	out[13] = dataset.Quantize(float32(merch.AvgAmount / MaxMerchantAvgAmount))
	return nil
}

func dayNumSlow(t time.Time) int {
	y, mo, d := t.Date()
	yPrev := y - 1
	days := yPrev*365 + yPrev/4 - yPrev/100 + yPrev/400
	days += monthDaysBefore[int(mo)-1] + d - 1
	if mo > time.February && isLeap(y) {
		days++
	}
	return days
}

// TestVectorizeFastMatchesSlow runs every request from test-data.json through
// both parsers and asserts the int16 outputs match.
func TestVectorizeFastMatchesSlow(t *testing.T) {
	f, err := os.Open("../../references/rinha-official/test/test-data.json")
	if err != nil {
		t.Skipf("test data unavailable (%v)", err)
		return
	}
	defer f.Close()

	var top struct {
		Entries []struct {
			Request json.RawMessage `json:"request"`
		} `json:"entries"`
	}
	if err := json.NewDecoder(f).Decode(&top); err != nil {
		t.Fatalf("decode top: %v", err)
	}

	for i, entry := range top.Entries {
		var slow, fast [dataset.Stride]int16
		if err := vectorizeSlow(entry.Request, &slow); err != nil {
			t.Fatalf("entry %d slow: %v", i, err)
		}
		if !VectorizeFast(entry.Request, &fast) {
			t.Fatalf("entry %d fast returned false: body=%s", i, entry.Request)
		}
		if slow != fast {
			t.Fatalf("entry %d:\n  slow=%v\n  fast=%v\n  body=%s", i, slow, fast, entry.Request)
		}
	}
	t.Logf("checked %d entries, all match", len(top.Entries))
}
