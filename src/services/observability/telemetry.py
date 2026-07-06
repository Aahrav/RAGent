"""OpenTelemetry Configuration.

This module initializes OpenTelemetry for distributed tracing.
By default, traces are exported to the console for local debugging.
"""
import os
import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME

try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    HAS_OTLP = True
except ImportError:
    HAS_OTLP = False

def init_telemetry(service_name: str = "ragent"):
    """Initialize OpenTelemetry tracer provider."""
    
    # Create a resource to identify this service in traces
    resource = Resource(attributes={
        SERVICE_NAME: service_name
    })

    # Set up the tracer provider
    provider = TracerProvider(resource=resource)
    
    # Export to Jaeger (via OTLP) if available, otherwise print to console
    if HAS_OTLP:
        endpoint = os.getenv("OTLP_ENDPOINT", "http://localhost:4317")
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        logging.info(f"OpenTelemetry using OTLPSpanExporter ({endpoint})")
    else:
        exporter = ConsoleSpanExporter()
        logging.info("OpenTelemetry using ConsoleSpanExporter (OTLP package not installed)")
        
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)
    
    # Set the global default tracer provider
    trace.set_tracer_provider(provider)
    
    logging.info(f"OpenTelemetry initialized for service: {service_name}")
