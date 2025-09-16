# Kubernetes Deployment

Este directorio contiene manifiestos de Kubernetes para desplegar WhatsApp RAG en un cluster.

## Arquitectura

- **Deployment**: 3 replicas con health checks
- **Service**: ClusterIP para comunicación interna  
- **Ingress**: Punto de entrada HTTPS con certificados automáticos
- **HPA**: Auto scaling horizontal basado en CPU/memoria
- **PVC**: Volúmenes persistentes para datos y logs
- **Secrets**: Gestión segura de tokens de API

## Despliegue Rápido

### Prerrequisitos

```bash
# Kubernetes cluster funcionando
kubectl cluster-info

# Ingress controller (ej. nginx)
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.4/deploy/static/provider/cloud/deploy.yaml

# Cert-manager para SSL automático
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.3/cert-manager.yaml
```

### Configuración

1. **Crear secretos con tokens reales:**

```bash
# Codificar tokens en base64
echo -n "tu_github_token" | base64

# Editar deployment.yaml y reemplazar:
# REPLACE_WITH_BASE64_ENCODED_TOKEN
```

2. **Configurar dominio:**

```bash
# Editar deployment.yaml y cambiar:
# whatsapp-rag.yourdomain.com
```

### Despliegue

```bash
# Aplicar manifiestos
kubectl apply -f deployment.yaml

# Verificar estado
kubectl get pods -l app=whatsapp-rag
kubectl get services
kubectl get ingress
```

## Usando Kustomize

Para entornos múltiples (dev/staging/prod):

### Base

```bash
# Usar kustomization.yaml
kubectl apply -k .
```

### Entornos

Crear directorios por entorno:

```bash
mkdir -p overlays/{development,staging,production}
```

**overlays/development/kustomization.yaml:**
```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: whatsapp-rag-dev

resources:
  - ../../

patchesStrategicMerge:
  - replica-count.yaml
  - resources.yaml

images:
  - name: whatsapp-rag
    newTag: dev-latest
```

**overlays/development/replica-count.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: whatsapp-rag
spec:
  replicas: 1
```

Desplegar:
```bash
kubectl apply -k overlays/development
kubectl apply -k overlays/production
```

## Helm Charts

### Estructura

```bash
mkdir -p helm/whatsapp-rag/{templates,charts}
```

**helm/whatsapp-rag/Chart.yaml:**
```yaml
apiVersion: v2
name: whatsapp-rag
description: WhatsApp RAG Analysis System
version: 1.0.0
appVersion: 1.0.0
```

**helm/whatsapp-rag/values.yaml:**
```yaml
image:
  repository: whatsapp-rag
  tag: latest
  pullPolicy: Always

replicaCount: 3

service:
  type: ClusterIP
  port: 80
  targetPort: 7860

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: whatsapp-rag.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: whatsapp-rag-tls
      hosts:
        - whatsapp-rag.example.com

resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "1000m"

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

persistence:
  enabled: true
  storageClass: "standard"
  accessMode: ReadWriteMany
  size: 5Gi

secrets:
  githubToken: ""
  openaiApiKey: ""
```

### Instalación con Helm

```bash
# Instalar
helm install whatsapp-rag ./helm/whatsapp-rag \
  --set secrets.githubToken=tu_token \
  --set ingress.hosts[0].host=tu-dominio.com

# Actualizar
helm upgrade whatsapp-rag ./helm/whatsapp-rag \
  --set image.tag=v1.1.0

# Desinstalar
helm uninstall whatsapp-rag
```

## Configuración Avanzada

### Recursos y Límites

```yaml
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
    # Para usar GPU (opcional)
    nvidia.com/gpu: 1
  limits:
    memory: "4Gi"
    cpu: "2000m"
    nvidia.com/gpu: 1
```

### Affinity y Tolerations

```yaml
# En deployment.yaml
spec:
  template:
    spec:
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - whatsapp-rag
              topologyKey: kubernetes.io/hostname
      tolerations:
      - key: "high-memory"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"
```

### Almacenamiento

#### NFS Storage Class

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: nfs-storage
provisioner: kubernetes.io/nfs
parameters:
  server: nfs-server.example.com
  path: /exports/whatsapp-rag
allowVolumeExpansion: true
```

#### S3 via CSI Driver

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: whatsapp-data-s3-pv
spec:
  capacity:
    storage: 100Gi
  accessModes:
    - ReadWriteMany
  csi:
    driver: s3.csi.aws.com
    volumeHandle: whatsapp-rag-bucket
    volumeAttributes:
      bucketName: whatsapp-rag-data
      region: us-east-1
```

## Monitoreo

### Prometheus Metrics

```yaml
# ServiceMonitor for Prometheus
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: whatsapp-rag-metrics
spec:
  selector:
    matchLabels:
      app: whatsapp-rag
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "WhatsApp RAG Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{job=\"whatsapp-rag\"}[5m])"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.95, http_request_duration_seconds_bucket{job=\"whatsapp-rag\"})"
          }
        ]
      }
    ]
  }
}
```

### Alertas

```yaml
# PrometheusRule
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: whatsapp-rag-alerts
spec:
  groups:
  - name: whatsapp-rag
    rules:
    - alert: HighCPUUsage
      expr: rate(container_cpu_usage_seconds_total{pod=~"whatsapp-rag-.*"}[5m]) > 0.8
      for: 5m
      annotations:
        summary: "High CPU usage detected"
    - alert: PodCrashLooping
      expr: rate(kube_pod_container_status_restarts_total{pod=~"whatsapp-rag-.*"}[15m]) > 0
      for: 5m
      annotations:
        summary: "Pod is crash looping"
```

## Seguridad

### Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: whatsapp-rag-netpol
spec:
  podSelector:
    matchLabels:
      app: whatsapp-rag
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 7860
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS para APIs externas
    - protocol: TCP
      port: 53   # DNS
    - protocol: UDP
      port: 53   # DNS
```

### Pod Security Context

```yaml
spec:
  template:
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1001
        runAsGroup: 1001
        fsGroup: 1001
      containers:
      - name: whatsapp-rag
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
```

### RBAC

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: whatsapp-rag-sa

---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: whatsapp-rag-role
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: whatsapp-rag-binding
subjects:
- kind: ServiceAccount
  name: whatsapp-rag-sa
roleRef:
  kind: Role
  name: whatsapp-rag-role
  apiGroup: rbac.authorization.k8s.io
```

## Troubleshooting

### Comandos Útiles

```bash
# Estado general
kubectl get pods,svc,ingress -l app=whatsapp-rag

# Logs
kubectl logs -l app=whatsapp-rag -f
kubectl logs deployment/whatsapp-rag --previous

# Debugging
kubectl describe pod whatsapp-rag-xxx
kubectl exec -it whatsapp-rag-xxx -- /bin/bash

# Eventos
kubectl get events --sort-by=.metadata.creationTimestamp

# Métricas de recursos
kubectl top pods -l app=whatsapp-rag
kubectl top nodes
```

### Problemas Comunes

1. **Pods no inician**: Verificar resources limits y node capacity
2. **Health checks fallan**: Revisar readiness/liveness probe settings  
3. **Sin acceso externo**: Verificar ingress controller y DNS
4. **Performance lenta**: Ajustar CPU/memoria o escalar horizontalmente
5. **Storage issues**: Verificar PVC status y storage class

### Debug de Red

```bash
# Test conectividad interna
kubectl run debug --image=nicolaka/netshoot -it --rm
# Dentro del pod:
nslookup whatsapp-rag-service
curl http://whatsapp-rag-service/

# Test desde external
kubectl port-forward svc/whatsapp-rag-service 8080:80
curl http://localhost:8080
```