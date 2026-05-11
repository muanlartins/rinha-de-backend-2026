# --- build stage ----------------------------------------------------------
FROM --platform=linux/amd64 golang:1.25-alpine AS builder
WORKDIR /src

COPY go.mod ./
RUN go mod download

COPY cmd/ cmd/
COPY internal/ internal/

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -trimpath -ldflags='-s -w' -o /out/api ./cmd/api

# --- runtime stage --------------------------------------------------------
FROM scratch
COPY --from=builder /out/api /api
ENTRYPOINT ["/api"]
