# --- build stage ----------------------------------------------------------
FROM --platform=linux/amd64 golang:1.25-alpine AS builder
WORKDIR /src

COPY go.mod go.sum ./
RUN go mod download

COPY cmd/ cmd/
COPY internal/ internal/

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -trimpath -ldflags='-s -w' -o /out/api ./cmd/api

# --- runtime stage --------------------------------------------------------
# Bake references.json.gz in at /resources/references.json.gz — the rinha test
# infra doesn't mount this; submissions must distribute it themselves. See
# the FAQ: "Pre-process the dataset during the container build."
FROM scratch
COPY --from=builder /out/api /api
COPY references/rinha-official/resources/references.json.gz /resources/references.json.gz
ENTRYPOINT ["/api"]
