# AWS Deployment

Este directorio contiene recursos para desplegar WhatsApp RAG en AWS usando ECS Fargate.

## Arquitectura

- **ECS Fargate**: Contenedores sin servidor para la aplicación
- **Application Load Balancer**: Distribuye tráfico y maneja SSL
- **Auto Scaling**: Escala automáticamente según el uso de CPU
- **S3**: Almacenamiento de archivos de WhatsApp
- **Secrets Manager**: Gestión segura de tokens de API
- **CloudWatch**: Monitoreo y logs

## Despliegue Rápido

### Prerrequisitos

```bash
# Instalar AWS CLI
pip install awscli

# Configurar credenciales AWS
aws configure

# Verificar acceso
aws sts get-caller-identity
```

### Despliegue Automático

```bash
# Hacer ejecutable el script (en systems Unix)
chmod +x deploy.sh

# Desplegar
./deploy.sh deploy
```

El script te pedirá el GitHub token si no está en `.env`.

### Comandos Disponibles

```bash
./deploy.sh deploy   # Despliega la aplicación
./deploy.sh destroy  # Elimina todo el stack
./deploy.sh status   # Muestra estado del stack
./deploy.sh logs     # Muestra logs en tiempo real
```

## Despliegue Manual

### 1. Crear Repositorio ECR

```bash
aws ecr create-repository --repository-name whatsapp-rag --region us-east-1
```

### 2. Construir y Subir Imagen

```bash
# Login a ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com

# Construir imagen
docker build -f ../docker/Dockerfile -t whatsapp-rag ../../

# Tag y push
docker tag whatsapp-rag:latest YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/whatsapp-rag:latest
docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/whatsapp-rag:latest
```

### 3. Desplegar CloudFormation

```bash
aws cloudformation deploy \
  --template-file cloudformation-template.yaml \
  --stack-name whatsapp-rag \
  --parameter-overrides GitHubToken=tu_token_aqui \
  --capabilities CAPABILITY_IAM \
  --region us-east-1
```

### 4. Subir Datos de Ejemplo

```bash
# Obtener nombre del bucket
BUCKET=$(aws cloudformation describe-stacks --stack-name whatsapp-rag --query 'Stacks[0].Outputs[?OutputKey==`DataBucketName`].OutputValue' --output text)

# Subir archivos
aws s3 sync ../../data s3://$BUCKET/data/
```

## Configuración

### Variables de Entorno

El stack configura automáticamente:

- `OPENAI_API_KEY`: Token de GitHub (desde Secrets Manager)
- `GITHUB_TOKEN`: Token de GitHub (desde Secrets Manager)  
- `OPENAI_API_BASE`: https://models.inference.ai.azure.com
- `MODEL_NAME`: gpt-4o-mini
- `GRADIO_SERVER_NAME`: 0.0.0.0
- `GRADIO_SERVER_PORT`: 7860

### Parámetros del Stack

| Parámetro | Descripción | Valor por Defecto |
|-----------|-------------|-------------------|
| `GitHubToken` | Token de GitHub para API | Requerido |
| `VpcCidr` | CIDR block para VPC | 10.0.0.0/16 |
| `DesiredCapacity` | Número de instancias | 2 |

### Recursos Creados

- **VPC**: Red privada virtual con 2 subnets públicas
- **ECS Cluster**: Cluster Fargate con auto scaling
- **Load Balancer**: ALB público con health checks
- **S3 Bucket**: Almacenamiento de datos con versionado
- **Secrets Manager**: Gestión segura de tokens
- **CloudWatch**: Logs y métricas

## Monitoreo

### CloudWatch Logs

```bash
# Ver logs en tiempo real
aws logs tail /ecs/whatsapp-rag --follow --region us-east-1

# Ver logs específicos
aws logs filter-log-events --log-group-name /ecs/whatsapp-rag --start-time $(date -d '1 hour ago' +%s)000
```

### Métricas

- CPU y memoria por servicio
- Número de tareas en ejecución
- Health checks del load balancer
- Latencia de respuesta

### Alertas CloudWatch

Ejemplo de alerta por alta CPU:

```bash
aws cloudwatch put-metric-alarm \
  --alarm-name "WhatsAppRAG-HighCPU" \
  --alarm-description "High CPU utilization" \
  --metric-name CPUUtilization \
  --namespace AWS/ECS \
  --statistic Average \
  --period 300 \
  --threshold 80 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2
```

## Seguridad

### Acceso a Datos

- Bucket S3 bloqueado para acceso público
- Acceso solo via IAM roles del ECS
- Versionado activado para auditoría

### Red

- Subnets públicas para load balancer únicamente
- Security groups restrictivos
- Tráfico HTTPS recomendado (requiere certificado SSL)

### Secretos

- Tokens almacenados en Secrets Manager
- Rotación automática disponible
- Acceso auditado via CloudTrail

## Costos Estimados

Para una configuración con 2 tareas Fargate (1 vCPU, 2GB RAM):

- **ECS Fargate**: ~$35/mes por tarea
- **Load Balancer**: ~$20/mes
- **S3**: <$5/mes (datos típicos)
- **Secrets Manager**: $0.40/mes por secreto
- **CloudWatch**: <$10/mes

**Total estimado**: ~$100-120/mes

### Optimización de Costos

- Usar FARGATE_SPOT para cargas no críticas
- Ajustar auto scaling según patrones de uso
- Configurar lifecycle policies en S3
- Usar reservas de capacidad para cargas predecibles

## SSL/HTTPS

Para habilitar HTTPS, modifica el listener del load balancer:

```yaml
# En cloudformation-template.yaml
Listener:
  Type: AWS::ElasticLoadBalancingV2::Listener
  Properties:
    LoadBalancerArn: !Ref LoadBalancer
    Port: 443
    Protocol: HTTPS
    Certificates:
      - CertificateArn: tu-certificado-arn
    DefaultActions:
      - Type: forward
        TargetGroupArn: !Ref TargetGroup
```

## Troubleshooting

### Errores Comunes

1. **Task no inicia**: Verificar logs en CloudWatch
2. **Health check falla**: Verificar puerto y ruta
3. **Sin acceso a secrets**: Revisar IAM permissions
4. **Load balancer timeout**: Ajustar health check settings

### Comandos de Diagnóstico

```bash
# Estado del servicio
aws ecs describe-services --cluster whatsapp-rag-cluster --services whatsapp-rag-service

# Tareas en ejecución
aws ecs list-tasks --cluster whatsapp-rag-cluster --service-name whatsapp-rag-service

# Health del load balancer
aws elbv2 describe-target-health --target-group-arn ARN_DEL_TARGET_GROUP
```