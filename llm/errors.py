class ModelError(Exception):
    "Models can raise this error, which will be displayed to the user"


class NeedsKeyException(ModelError):
    "Model needs an API key which has not been provided"


class ProviderAPIError(ModelError):
    "Error returned by the provider API"

    def __init__(self, message: str, status_code: int = None, response: dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class ProviderRateLimitError(ProviderAPIError):
    "Rate limit exceeded for the provider"


class ProviderAuthenticationError(ProviderAPIError):
    "Authentication failed with the provider"


class ProviderTimeoutError(ModelError):
    "Request to provider timed out"


class ProviderConnectionError(ModelError):
    "Connection error with the provider"


class ProviderResponseError(ModelError):
    "Unexpected response format from provider"

    def __init__(self, message: str, response: any = None):
        super().__init__(message)
        self.response = response


class ProviderConfigurationError(ModelError):
    "Invalid provider configuration"


class ProviderUnsupportedOperationError(ModelError):
    "Operation not supported by this provider"


class ProviderToolCallError(ModelError):
    "Error during tool call processing"
