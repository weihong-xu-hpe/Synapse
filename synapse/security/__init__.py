"""Security package for Synapse."""

from synapse.security.sanitization import (
	CREDENTIAL_PATTERN_DEFINITIONS,
	DEFAULT_PATTERN_DEFINITIONS,
	RedactionEngine,
	RedactionPattern,
	RedactionResult,
	build_default_patterns,
	build_redaction_patterns,
	purge_expired_archive_files,
	scrub_credentials,
	strip_lone_surrogates,
)

__all__ = [
	"CREDENTIAL_PATTERN_DEFINITIONS",
	"DEFAULT_PATTERN_DEFINITIONS",
	"RedactionEngine",
	"RedactionPattern",
	"RedactionResult",
	"build_default_patterns",
	"build_redaction_patterns",
	"purge_expired_archive_files",
	"scrub_credentials",
	"strip_lone_surrogates",
]
