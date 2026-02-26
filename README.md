# DataApps

A collection of data engineering and ML pipeline experiments.

## Projects

### [SimplePipeline](simplePipeline/)

A multi-model data classification pipeline that detects PII across documents using a layered approach — regex, NER, a fine-tuned secret-detection SLM, and a general LLM. Each layer is progressively more capable and expensive, with confidence-based routing to minimize cost while maximizing accuracy.

See [SimplePipeline README](simplePipeline/README.md) for details.
