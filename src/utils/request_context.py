from contextvars import ContextVar

# This variable holds the request ID for the current async task.
# It is thread-safe and async-safe.
_request_id_ctx_var: ContextVar[str] = ContextVar("request_id", default="")
_user_id_ctx_var: ContextVar[str] = ContextVar("user_id", default="anonymous")

def get_request_id() -> str:
    """Retrieve the request ID for the current request context.
    
    Any file can call this to tag its logs,
    without needing the ID passed as a function argument.
    """
    return _request_id_ctx_var.get()

def set_request_id(req_id: str):
    """Set the request ID for the current context."""
    return _request_id_ctx_var.set(req_id)

def reset_request_id(token):
    """Reset the request ID."""
    _request_id_ctx_var.reset(token)

def get_user_id() -> str:
    """Retrieve the user ID for the current request context."""
    return _user_id_ctx_var.get()

def set_user_id(user_id: str):
    """Set the user ID for the current context."""
    return _user_id_ctx_var.set(user_id)

def reset_user_id(token):
    """Reset the user ID."""
    _user_id_ctx_var.reset(token)
