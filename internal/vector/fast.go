package vector

import (
	"bytes"

	"github.com/muanlartins/rinha-de-backend-2026/internal/dataset"
)

var (
	kTransAmount = []byte(`"amount":`)
	kTransInst   = []byte(`"installments":`)
	kTransReq    = []byte(`"requested_at":"`)
	kCustAvg     = []byte(`"avg_amount":`)
	kCustTx24    = []byte(`"tx_count_24h":`)
	kCustKnown   = []byte(`"known_merchants":[`)
	kMerchObj    = []byte(`"merchant":`)
	kMerchID     = []byte(`"id":"`)
	kMerchMCC    = []byte(`"mcc":"`)
	kMerchAvg    = []byte(`"avg_amount":`)
	kTermOnline  = []byte(`"is_online":`)
	kTermPresent = []byte(`"card_present":`)
	kTermKmHome  = []byte(`"km_from_home":`)
	kLastTx      = []byte(`"last_transaction":`)
	kLastTS      = []byte(`"timestamp":"`)
	kLastKM      = []byte(`"km_from_current":`)
)

var mccRiskTable [10000]int16

func init() {
	defaultQ := dataset.Quantize(DefaultMccRisk)
	for i := range mccRiskTable {
		mccRiskTable[i] = defaultQ
	}
	for codeStr, risk := range MccRisk {
		code := 0
		for i := 0; i < len(codeStr); i++ {
			code = code*10 + int(codeStr[i]-'0')
		}
		mccRiskTable[code] = dataset.Quantize(risk)
	}
}

// VectorizeFast parses body and fills out with int16 quantized dims. Returns
// false on malformed payloads — caller must avoid producing an HTTP 5xx
// (failure weight 5 in E; misclassification is at most 3).
func VectorizeFast(body []byte, out *[dataset.Stride]int16) bool {
	p := 0

	idx := bytes.Index(body[p:], kTransAmount)
	if idx < 0 {
		return false
	}
	p += idx + len(kTransAmount)
	amount, next, ok := parseFloat(body, p)
	if !ok {
		return false
	}
	out[0] = quantClamp01(amount / MaxAmount)
	p = next

	idx = bytes.Index(body[p:], kTransInst)
	if idx < 0 {
		return false
	}
	p += idx + len(kTransInst)
	installments, next, ok := parseFloat(body, p)
	if !ok {
		return false
	}
	out[1] = quantClamp01(installments / MaxInstallments)
	p = next

	idx = bytes.Index(body[p:], kTransReq)
	if idx < 0 {
		return false
	}
	reqAt := p + idx + len(kTransReq)
	if reqAt+15 >= len(body) {
		return false
	}
	reqYear := fourDigits(body, reqAt)
	reqMonth := twoDigits(body, reqAt+5)
	reqDay := twoDigits(body, reqAt+8)
	reqHour := twoDigits(body, reqAt+11)
	reqMin := twoDigits(body, reqAt+14)
	reqDayNum := dayNumberFromYMD(reqYear, reqMonth, reqDay)
	reqMinutes := reqDayNum*1440 + reqHour*60 + reqMin

	out[3] = quantClamp01(float64(reqHour) / 23.0)
	out[4] = quantClamp01(float64(reqDayNum%7) / 6.0)
	p = reqAt + 20

	idx = bytes.Index(body[p:], kCustAvg)
	if idx < 0 {
		return false
	}
	p += idx + len(kCustAvg)
	custAvg, next, ok := parseFloat(body, p)
	if !ok {
		return false
	}
	if custAvg > 0 {
		out[2] = quantClamp01((amount / custAvg) / AmountVsAvgRatio)
	} else {
		out[2] = dataset.QuantScale
	}
	p = next

	idx = bytes.Index(body[p:], kCustTx24)
	if idx < 0 {
		return false
	}
	p += idx + len(kCustTx24)
	tx24, next, ok := parseFloat(body, p)
	if !ok {
		return false
	}
	out[8] = quantClamp01(tx24 / MaxTxCount24h)
	pAfterTx24 := next

	idx = bytes.Index(body[pAfterTx24:], kMerchObj)
	if idx < 0 {
		return false
	}
	mp := pAfterTx24 + idx
	idx = bytes.Index(body[mp:], kMerchID)
	if idx < 0 {
		return false
	}
	merchIDStart := mp + idx + len(kMerchID)
	merchIDEnd := merchIDStart
	for merchIDEnd < len(body) && body[merchIDEnd] != '"' {
		merchIDEnd++
	}
	if merchIDEnd >= len(body) {
		return false
	}
	merchIDLen := merchIDEnd - merchIDStart

	idx = bytes.Index(body[merchIDEnd:], kMerchMCC)
	if idx < 0 {
		return false
	}
	mccStart := merchIDEnd + idx + len(kMerchMCC)
	if mccStart+3 >= len(body) {
		return false
	}
	mcc := int(body[mccStart]-'0')*1000 +
		int(body[mccStart+1]-'0')*100 +
		int(body[mccStart+2]-'0')*10 +
		int(body[mccStart+3]-'0')
	if mcc < 0 || mcc >= 10000 {
		out[12] = dataset.Quantize(DefaultMccRisk)
	} else {
		out[12] = mccRiskTable[mcc]
	}

	idx = bytes.Index(body[mccStart+4:], kMerchAvg)
	if idx < 0 {
		return false
	}
	mavgStart := mccStart + 4 + idx + len(kMerchAvg)
	merchAvg, next, ok := parseFloat(body, mavgStart)
	if !ok {
		return false
	}
	out[13] = quantClamp01(merchAvg / MaxMerchantAvgAmount)
	pAfterMerch := next

	idx = bytes.Index(body[pAfterTx24:], kCustKnown)
	if idx < 0 {
		return false
	}
	knownStart := pAfterTx24 + idx + len(kCustKnown)
	knownEnd := bytes.IndexByte(body[knownStart:], ']')
	if knownEnd < 0 {
		return false
	}
	knownEnd += knownStart
	found := false
	i := knownStart
	for i < knownEnd {
		if body[i] == '"' {
			itemStart := i + 1
			itemEnd := itemStart
			for itemEnd < knownEnd && body[itemEnd] != '"' {
				itemEnd++
			}
			if itemEnd >= knownEnd {
				break
			}
			if itemEnd-itemStart == merchIDLen &&
				bytes.Equal(body[itemStart:itemEnd], body[merchIDStart:merchIDEnd]) {
				found = true
				break
			}
			i = itemEnd + 1
			continue
		}
		i++
	}
	if found {
		out[11] = 0
	} else {
		out[11] = dataset.QuantScale
	}

	idx = bytes.Index(body[pAfterMerch:], kTermOnline)
	if idx < 0 {
		return false
	}
	tp := pAfterMerch + idx + len(kTermOnline)
	if tp >= len(body) {
		return false
	}
	if body[tp] == 't' {
		out[9] = dataset.QuantScale
	} else {
		out[9] = 0
	}

	idx = bytes.Index(body[tp:], kTermPresent)
	if idx < 0 {
		return false
	}
	tp += idx + len(kTermPresent)
	if tp >= len(body) {
		return false
	}
	if body[tp] == 't' {
		out[10] = dataset.QuantScale
	} else {
		out[10] = 0
	}

	idx = bytes.Index(body[tp:], kTermKmHome)
	if idx < 0 {
		return false
	}
	tp += idx + len(kTermKmHome)
	kmHome, next, ok := parseFloat(body, tp)
	if !ok {
		return false
	}
	out[7] = quantClamp01(kmHome / MaxKm)
	pAfterTerm := next

	idx = bytes.Index(body[pAfterTerm:], kLastTx)
	if idx < 0 {
		return false
	}
	lp := pAfterTerm + idx + len(kLastTx)
	for lp < len(body) && (body[lp] == ' ' || body[lp] == '\t' || body[lp] == '\n' || body[lp] == '\r') {
		lp++
	}
	if lp >= len(body) {
		return false
	}
	if body[lp] == 'n' {
		out[5] = dataset.SentinelInt
		out[6] = dataset.SentinelInt
		return true
	}

	idx = bytes.Index(body[lp:], kLastTS)
	if idx < 0 {
		return false
	}
	tsStart := lp + idx + len(kLastTS)
	if tsStart+15 >= len(body) {
		return false
	}
	lastYear := fourDigits(body, tsStart)
	lastMonth := twoDigits(body, tsStart+5)
	lastDay := twoDigits(body, tsStart+8)
	lastHour := twoDigits(body, tsStart+11)
	lastMin := twoDigits(body, tsStart+14)
	lastDayNum := dayNumberFromYMD(lastYear, lastMonth, lastDay)
	lastMinutes := lastDayNum*1440 + lastHour*60 + lastMin
	out[5] = quantClamp01(float64(reqMinutes-lastMinutes) / MaxMinutes)

	idx = bytes.Index(body[tsStart:], kLastKM)
	if idx < 0 {
		return false
	}
	kmStart := tsStart + idx + len(kLastKM)
	kmCurrent, _, ok := parseFloat(body, kmStart)
	if !ok {
		return false
	}
	out[6] = quantClamp01(kmCurrent / MaxKm)
	return true
}

var pow10 = [16]float64{
	1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7,
	1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
}

// parseFloat accumulates the fractional part as an integer and divides once
// at the end — `frac *= 0.1` per digit compounds float rounding error.
func parseFloat(body []byte, i int) (float64, int, bool) {
	for i < len(body) && (body[i] == ' ' || body[i] == '\t' || body[i] == '\n' || body[i] == '\r') {
		i++
	}
	if i >= len(body) {
		return 0, i, false
	}
	neg := false
	if body[i] == '-' {
		neg = true
		i++
	}
	if i >= len(body) || body[i] < '0' || body[i] > '9' {
		return 0, i, false
	}
	var intPart int64
	for i < len(body) && body[i] >= '0' && body[i] <= '9' {
		intPart = intPart*10 + int64(body[i]-'0')
		i++
	}
	val := float64(intPart)
	if i < len(body) && body[i] == '.' {
		i++
		if i >= len(body) || body[i] < '0' || body[i] > '9' {
			return 0, i, false
		}
		var fracPart int64
		fracDigits := 0
		for i < len(body) && body[i] >= '0' && body[i] <= '9' {
			fracPart = fracPart*10 + int64(body[i]-'0')
			fracDigits++
			i++
		}
		if fracDigits >= len(pow10) {
			fracDigits = len(pow10) - 1
		}
		val += float64(fracPart) / pow10[fracDigits]
	}
	if neg {
		val = -val
	}
	return val, i, true
}

func twoDigits(b []byte, i int) int {
	return int(b[i]-'0')*10 + int(b[i+1]-'0')
}

func fourDigits(b []byte, i int) int {
	return int(b[i]-'0')*1000 + int(b[i+1]-'0')*100 + int(b[i+2]-'0')*10 + int(b[i+3]-'0')
}

// dayNumberFromYMD: Jan 1 of year 1 is day 0 (proleptic Gregorian, Monday).
func dayNumberFromYMD(y, m, d int) int {
	yPrev := y - 1
	days := yPrev*365 + yPrev/4 - yPrev/100 + yPrev/400
	days += monthDaysBefore[m-1] + d - 1
	if m > 2 && isLeap(y) {
		days++
	}
	return days
}

// quantClamp01: the float32 round-trip must match dataset.Quantize, which
// quantizes the reference set. Staying in float64 here diverges by 1 unit
// near a quantization boundary.
func quantClamp01(x float64) int16 {
	v := float32(x)
	if v <= 0 {
		return 0
	}
	if v >= 1 {
		return dataset.QuantScale
	}
	return int16(v*float32(dataset.QuantScale) + 0.5)
}
