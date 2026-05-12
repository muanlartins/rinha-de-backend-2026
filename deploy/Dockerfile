# --- build stage ----------------------------------------------------------
FROM --platform=linux/amd64 golang:1.25-alpine AS builder
WORKDIR /src

COPY go.mod go.sum ./
RUN go mod download

COPY cmd/ cmd/
COPY internal/ internal/

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -trimpath -ldflags='-s -w' -o /out/api ./cmd/api && \
    CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
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
