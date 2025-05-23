# options
cmd := ../clusters/abci3.py
job_ops := 

# general rules
%: status

.stage%.done:
	touch $@
	@echo "\033[1;34mStage$(shell basename $*) finished\033[0m"

status:
	@echo ================================================================
	@$(foreach var,$(PRINT_VARIABLES),echo $(var): $($(var));)
	@echo ================================================================
