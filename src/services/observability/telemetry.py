"""OpenTelemetry Configuration.

This module initializes OpenTelemetry for distributed tracing.
By default, traces are exported to the console for local debugging.
"""
import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME

def init_telemetry(service_name: str = "ragent"):
    """Initialize OpenTelemetry tracer provider."""
    
    # Create a resource to identify this service in traces
    resource = Resource(attributes={
        SERVICE_NAME: service_name
    })

    # Set up the tracer provider
    provider = TracerProvider(resource=resource)
    
    # We use a ConsoleSpanExporter to print traces to stdout.
    # In production, you would replace this with OTLPSpanExporter to send to Jaeger/Zipkin.
    processor = BatchSpanProcessor(ConsoleSpanExporter())
    provider.add_span_processor(processor)
    
    # Set the global default tracer provider
    trace.set_tracer_provider(provider)
    
    logging.info(f"OpenTelemetry initialized for service: {service_name}")
