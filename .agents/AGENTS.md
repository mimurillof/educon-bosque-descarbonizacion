# Reglas del Proyecto (Customizations)

### Reglas de Formateo de Contenido en HTML
- **Evitar Sintaxis Markdown**: Jamás utilices marcas literales de Markdown (como `**` para negrita, `*` para cursiva o `_`) dentro de etiquetas HTML (`<p>`, `<li>`, `span`, etc.). Los navegadores no interpretan Markdown de forma nativa. Usa siempre etiquetas HTML estándar como `<strong>`, `<b>`, `<em>` o `<i>`.

### Directrices de Modelización EdTech y Validación B2B
- **Funnel Estocástico de 3 Etapas**: Al modelar embudos de educación continua o EdTech B2B, evita conversiones simplificadas de dos etapas o factores de escala arbitrarios. Emplea el funnel estándar histórico de la industria:
  1. Tráfico a Lead (3% - 6%)
  2. Lead a Entrevista Comercial (15% - 25%)
  3. Entrevista a Matrícula (8% - 15%)
  Representa estas tasas mediante distribuciones Beta independientes y propaga la incertidumbre secuencialmente mediante muestreos Binomiales.
- **Deducción de Costos Publicitarios**: La simulación de utilidad financiera neta (VPN) en campañas de adquisición debe descontar el costo real de captación del tráfico: $\text{Costo de Marketing} = \text{Tráfico} \times \text{CPC Ponderado}$.
- **Análisis de Precios Hedónicos**: En modelos de regresión hedónica para fijación de precios educativos, interpreta siempre el intercepto ($\beta_0$) como la prima o penalización por prestigio/barrera de entrada del clúster institucional no élite.
