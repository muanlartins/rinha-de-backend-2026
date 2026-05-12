package vector

// Normalization constants from resources/normalization.json.
const (
	MaxAmount            = 10000.0
	MaxInstallments      = 12.0
	AmountVsAvgRatio     = 10.0
	MaxMinutes           = 1440.0
	MaxKm                = 1000.0
	MaxTxCount24h        = 20.0
	MaxMerchantAvgAmount = 10000.0
)

// MccRisk mirrors resources/mcc_risk.json. Codes not listed use DefaultMccRisk.
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

var monthDaysBefore = [12]int{0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334}

func isLeap(y int) bool {
	return (y%4 == 0 && y%100 != 0) || y%400 == 0
}
