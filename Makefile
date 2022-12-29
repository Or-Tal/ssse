INTEG=_DORA_TEST_PATH="/tmp/magma_$(USER)" python3 -m dora -v run --clear device=cpu dataset.num_workers=0 optim.epochs=1 dataset.train.num_samples=10 dataset.valid.num_samples=10 dataset.evaluate.num_samples=10 dataset.generate.num_samples=2
INTEG_LM=$(INTEG) solver=lm/debug dset=audio/debug compression_model_checkpoint=e199ec82 \
		sample_rate=16000 transformer_lm.n_q=2 transformer_lm.dim=16 checkpoint.save_last=false

default: linter tests test_integ

linter:
	flake8 magma && mypy magma
	flake8 tests && mypy tests
	flake8 scripts & mypy scripts

tests:
	coverage run -m pytest tests

test_integ:
	$(INTEG) solver=compression/debug rvq.n_q=2 checkpoint.save_last=true   # SIG is e199ec82
	$(INTEG_LM)
	$(INTEG_LM) transformer_lm.residual_balancer_attn=0.1  \
		transformer_lm.spectral_norm_ff_iters=4 transformer_lm.norm_first=true

# Run this one for all integration tests (will be slower)
test_integ_extra:
	$(INTEG) solver=dummy/dummy_v1
	$(INTEG) solver=compression/debug rvq.n_q=2 balancer.balance_grads=true \
		checkpoint.save_last=false


.PHONY: linter tests test_integ test_integ_extra
