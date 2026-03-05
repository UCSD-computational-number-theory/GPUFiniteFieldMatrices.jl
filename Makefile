TARGET ?= .
FILE ?=

.PHONY: quality quality-target quality-aqua quality-jet quality-staticlint quality-alloccheck quality-dispatchdoctor quality-formatter fmt fmt-check fmt-file

quality: quality-aqua quality-jet quality-staticlint quality-alloccheck quality-dispatchdoctor quality-formatter

quality-target: quality

quality-aqua:
	julia --project=. test/Quality/aqua.jl "$(TARGET)"

quality-jet:
	julia --project=. test/Quality/jet.jl "$(TARGET)"

quality-staticlint:
	julia --project=. test/Quality/staticlint.jl "$(TARGET)"

quality-alloccheck:
	julia --project=. test/Quality/alloccheck.jl "$(TARGET)"

quality-dispatchdoctor:
	julia --project=. test/Quality/dispatchdoctor.jl "$(TARGET)"

quality-formatter:
	julia --project=. test/Quality/formatter.jl check "$(TARGET)"

fmt:
	julia --project=. test/Quality/formatter.jl write "$(TARGET)"

fmt-check:
	julia --project=. test/Quality/formatter.jl check "$(TARGET)"

fmt-file:
	@if [ -z "$(FILE)" ]; then echo "Usage: make fmt-file FILE=path/to/file.jl"; exit 2; fi
	julia --project=. test/Quality/formatter.jl write "$(FILE)"
