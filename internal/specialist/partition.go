// Package specialist implements Phase 36's index: partition references
// by an 8-bit categorical key (no centroid scoring at query time), then
// build a balanced KD-tree inside each partition for bbox-LB pruning at
// every level.
//
// Design source: fksegundo's "specialist partitioning" approach. We
// adopt the partition-key scheme (purely derived from the dataset's
// categorical structure) and the KD-tree-per-partition layout. We
// explicitly do NOT adopt the corrected_fraud_count hardcoded lookup
// table — that's a test-set lookup and violates the challenge rules.
//
// Why this beats IVF for our workload:
//
//   - O(1) partition selection. IVF computes K=4096 centroid distances
//     on every query (~4.5 µs after asm); specialist does 8 bit checks.
//   - Tighter bbox-LB pruning. KD-tree subdivides each partition with
//     per-node sub-bboxes; IVF has one bbox per cluster.
//   - Categorical fidelity. 7 of our 14 dims are essentially binary
//     (sentinels, online/offline, mcc-bucketed). Bitwise partitioning
//     respects that exactly; k-means smooths over it.
package specialist

import "github.com/muanlartins/rinha-de-backend-2026/internal/dataset"

// PartitionKey computes the 8-bit categorical partition key for a
// vector. Bit layout (verified against fksegundo's reference impl):
//
//   bit 0: has_last_tx        (vector[5] >= 0)
//   bit 1: is_online          (vector[9]  > 0)
//   bit 2: card_present       (vector[10] > 0)
//   bit 3: unknown_merchant   (vector[11] > 0)
//   bits 4-5: mcc bucket      ({0..3} from vector[12])
//   bit 6: amount > 5× avg    (vector[2] > 4096)
//   bit 7: tx_count_24h > 5   (vector[8] > 2048)
//
// Total: 8 bits → up to 256 partitions. The dataset typically populates
// 50-150 of those (many bit combos are correlated).
//
// CRITICAL: this function is called once per query. Keep it branchless
// and inlinable; avoid switch where bitwise gives the same result.
func PartitionKey(v *[dataset.Dims]int16) uint32 {
	var key uint32

	// 4 binary bits.
	if v[5] >= 0 {
		key |= 1 << 0
	}
	if v[9] > 0 {
		key |= 1 << 1
	}
	if v[10] > 0 {
		key |= 1 << 2
	}
	if v[11] > 0 {
		key |= 1 << 3
	}

	// 2-bit mcc bucket. At dataset.QuantScale=10000 our int16 mcc range
	// is [0..10000]; we want roughly equal-population quartiles.
	// fksegundo uses 2048 boundaries (suggests their scale was 8192-ish
	// — actually checking their code: they quantize differently. We
	// adjust to our quantization: 10000 / 4 = 2500 per quartile.)
	mcc := v[12]
	var mccBucket uint32
	switch {
	case mcc <= 2499:
		mccBucket = 0
	case mcc <= 4999:
		mccBucket = 1
	case mcc <= 7499:
		mccBucket = 2
	default:
		mccBucket = 3
	}
	key |= mccBucket << 4

	// amount/avg ratio threshold. Our quantization at scale 10000 means
	// ratio=0.5 → int16=5000. fksegundo used 4096 (≈0.5 at their scale).
	if v[2] > 5000 {
		key |= 1 << 6
	}

	// tx_count_24h threshold. ratio=0.25 → int16=2500.
	if v[8] > 2500 {
		key |= 1 << 7
	}

	return key
}

// MaxPartitions caps the partition directory size. With 8 bits the
// theoretical maximum is 256; we round up to 384 for a little
// headroom against future bit additions and to match a multiple of
// 64 for bitmap alignment in scratch buffers.
const MaxPartitions = 384
