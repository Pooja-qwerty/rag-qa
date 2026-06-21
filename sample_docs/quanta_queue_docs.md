# Quanta Queue — Engineering Documentation

Quanta Queue is a distributed task queue for running background jobs at scale.
This document covers installation, configuration, retry behavior, rate limits,
authentication, and error handling.

## Overview

Quanta Queue accepts jobs over HTTP or gRPC, persists them to a write-ahead log,
and dispatches them to worker pools. Each job is assigned a unique 26-character
ULID. Jobs move through the states: `queued` -> `running` -> `succeeded` or
`failed`. A job that exhausts its retries moves to the `dead` state and is sent
to the dead-letter queue (DLQ).

## Installation

Install the client library with pip:

    pip install quanta-queue

The server runs as a Docker container. The minimum supported version is 2.4.0.
You must run a PostgreSQL 14+ instance for the write-ahead log and a Redis 7+
instance for the dispatch buffer. Quanta will refuse to start if it cannot reach
both within 30 seconds of boot.

## Configuration

Configuration is read from `quanta.yaml`. The most important fields are:

- `worker_concurrency`: number of jobs a single worker runs in parallel.
  Default is 8. Setting this above 32 is not recommended because the Redis
  dispatch buffer becomes the bottleneck.
- `job_timeout_seconds`: how long a single job may run before it is killed and
  marked failed. Default is 300 (5 minutes). The maximum allowed value is 3600.
- `visibility_timeout_seconds`: how long a claimed job is hidden from other
  workers. Default is 600. This must always be larger than `job_timeout_seconds`,
  or two workers may pick up the same job.
- `dlq_enabled`: whether failed jobs are sent to the dead-letter queue. Default
  is true.

## Retry Logic

By default, a failed job is retried up to 5 times. Retries use exponential
backoff with jitter. The delay before retry number `n` is computed as:

    delay = min(base_delay * (2 ** n) + random_jitter, max_delay)

The default `base_delay` is 2 seconds and the default `max_delay` is 300 seconds.
Jitter is a random value between 0 and 1 second, added to avoid the thundering
herd problem where many jobs retry at the exact same moment.

To configure retries per job, set the `max_retries` field when enqueuing. Setting
`max_retries` to 0 disables retries entirely, so the job goes straight to the DLQ
on its first failure. Quanta does not retry jobs that fail with a `FATAL` error
code regardless of the `max_retries` setting.

## Rate Limits

The HTTP enqueue endpoint is rate limited to 2,000 requests per second per API
key. Exceeding this returns HTTP 429 with a `Retry-After` header indicating how
many milliseconds to wait. The gRPC endpoint is not rate limited but is bounded
by `worker_concurrency` on the server side.

Rate limits are enforced using a token bucket with a burst capacity of 4,000.
This means a client may briefly send up to 4,000 requests in a single burst
before the steady-state limit of 2,000/second applies.

## Authentication

All requests require an API key passed in the `Authorization` header as
`Bearer <key>`. API keys are scoped to a single namespace. A key with the
`admin` scope can enqueue to any namespace; a key with the `producer` scope can
only enqueue to its own namespace and cannot read job results.

API keys do not expire automatically. To rotate a key, create a new one in the
dashboard and revoke the old one. Revocation takes effect within 10 seconds
across all nodes.

## Error Codes

- `RETRYABLE`: a transient failure (for example a network timeout). The job will
  be retried according to its retry policy.
- `FATAL`: a permanent failure (for example invalid job payload). The job is
  never retried and goes straight to the DLQ.
- `TIMEOUT`: the job exceeded `job_timeout_seconds`. Treated as retryable.
- `THROTTLED`: returned as HTTP 429 when the rate limit is exceeded.

## Monitoring

Quanta exposes Prometheus metrics on port 9090 at `/metrics`. Key metrics include
`quanta_jobs_queued_total`, `quanta_jobs_failed_total`, and
`quanta_dlq_depth`. A rising `quanta_dlq_depth` usually indicates a bug in a job
handler rather than an infrastructure problem.
