# --- build stage ----------------------------------------------------------
FROM --platform=linux/amd64 golang:1.25-alpine AS builder
WORKDIR /src

COPY go.mod go.sum ./
RUN go mod download

COPY cmd/ cmd/
COPY internal/ internal/

# GOAMD64=v3 targets Haswell baseline (SSE4.2, AVX, AVX2, FMA3, BMI1/2).
# The Mac Mini Late 2014 test env is a 2.6 GHz Haswell, so v3 is safe.
# Default v1 leaves the compiler emitting baseline-x86_64 code that doesn't
# use even SSE4.2 — a free 2-8% on scalar hot loops to switch.
#
# -pgo=auto picks up cmd/api/default.pgo if present, otherwise no-ops.
# We capture default.pgo locally from a bench run and check it in.
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 GOAMD64=v3 \
    go build -trimpath -pgo=auto -ldflags='-s -w' -o /out/api ./cmd/api && \
    CGO_ENABLED=0 GOOS=linux GOARCH=amd64 GOAMD64=v3 \
    go build -trimpath -o /out/build-index ./cmd/build-index

# Pre-build the IVF index from references.json.gz so the runtime container
# starts in milliseconds instead of running k-means at boot (which times out
# the rinha-bot health check on the Mac Mini).
COPY references/rinha-official/resources/references.json.gz /tmp/references.json.gz
RUN mkdir -p /out/resources && \
    /out/build-index /tmp/references.json.gz /out/resources/index.bin && \
    rm /tmp/references.json.gz

# --- runtime stage --------------------------------------------------------
FROM scratch
COPY --from=builder /out/api /api
COPY --from=builder /out/resources/index.bin /resources/index.bin
ENTRYPOINT ["/api"]
