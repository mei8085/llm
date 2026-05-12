class ModelError(Exception):
    "Models can raise this error, which will be displayed to the user"


class NeedsKeyException(ModelError):
    "Model needs an API key which has not been provided"


class ResourceLimitExceeded(ModelError):
    "Raised when a plugin exceeds its declared resource limits"

    def __init__(self, message, resource_type=None, limit=None, usage=None):
        super().__init__(message)
        self.resource_type = resource_type
        self.limit = limit
        self.usage = usage
