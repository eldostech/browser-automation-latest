# Running this on EKS

Two Deployments, one queue, one database. The API serves HTTP and the workers
run batches; they are the same image with a different command, because they are
the same program with a different entry point.

```
        ALB
         │
    ┌────▼─────┐        ┌──────────────┐
    │   api    │        │   worker     │  ← KEDA scales this on queue depth
    │ (2+ pods)│        │ (0..N pods)  │
    └────┬─────┘        └──────┬───────┘
         │                     │
         └────────┬────────────┘
                  ▼
        Aurora PostgreSQL (pgvector)
                  │
                  ▼
              S3 (artifacts)
```

## What is deliberately not here

**Recording.** `playwright codegen` opens a headed browser and needs a display,
which a pod does not have. `RECORDER_ENABLED=false` is baked into the image so
the endpoints answer 501 with a sentence about displays, rather than a spawn
failing with an X11 error nobody can act on. Workflows are recorded on a laptop
and run here.

**A separate broker.** The queue is a Postgres table. That is not a shortcut:
enqueueing a batch and writing the rows it is about happen in *one transaction*,
so there is never a queued batch that no worker will claim. SQS cannot offer
that without an outbox table, which is to say without reimplementing this. KEDA
scales on either; its PostgreSQL scaler takes a query, and ours is one
`COUNT(*)`.

**Horizontal pod autoscaling on the API.** It is a request/response service in
front of a database; scale it on CPU if you need to, with the usual HPA. The
interesting scaling problem is the workers, and that is what KEDA is here for.

## Before you apply anything

1. **An Aurora PostgreSQL cluster with pgvector.** `CREATE EXTENSION vector`
   runs inside the migration, but the extension has to be *available* on the
   server first. Aurora ships it; check with
   `SELECT * FROM pg_available_extensions WHERE name = 'vector'`.
2. **An S3 bucket** for artifacts, and an IAM role the pods can assume (IRSA)
   with `s3:PutObject`, `s3:GetObject` on it plus `bedrock:InvokeModel` and
   `bedrock:InvokeModelWithResponseStream` for healing and embeddings.
3. **A secret** holding the database password and `CREDENTIALS_KEY`:
   ```bash
   kubectl create secret generic browser-automation \
     --from-literal=DB_PASSWORD='…' \
     --from-literal=CREDENTIALS_KEY="$(python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')"
   ```
   `CREDENTIALS_KEY` encrypts every stored site login. Losing it makes every
   saved credential unreadable; leaking it makes them all readable. It belongs
   in Secrets Manager with a rotation story, and the fact that this is a static
   key is the outstanding item **A4** in `TODO.md`.

## Applying

```bash
kubectl apply -f deploy/k8s/namespace.yaml
kubectl apply -f deploy/k8s/configmap.yaml
kubectl apply -f deploy/k8s/migrate-job.yaml   # wait for this
kubectl apply -f deploy/k8s/api.yaml
kubectl apply -f deploy/k8s/worker.yaml
kubectl apply -f deploy/k8s/keda.yaml
```

The migration is a Job rather than an init container on purpose: two API pods
starting together would otherwise both try to migrate, and Alembic's advisory
lock turns that into one waiting on the other rather than a failure — but a
Job makes "has the schema been upgraded" a thing you can look at, and a failed
upgrade stops the rollout instead of crash-looping every pod.

## Scaling behaviour

KEDA polls `COUNT(*) FROM jobs WHERE status = 'queued'` and runs one worker per
queued batch, to a ceiling. `minReplicaCount: 0` means an idle cluster runs no
workers at all, which matters because each one holds a Chromium.

A worker is not interruptible mid-batch without losing the row in flight, so
`cooldownPeriod` is generous and the pods get a long `terminationGracePeriod`.
The queue's lease makes this recoverable rather than merely polite: a worker
killed anyway releases its job when the lease expires, and another picks it up.
