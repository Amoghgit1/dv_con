# =============================================================
#  Makefile – Parallel BRAM-Based Affinity Scorer Simulation
#  DVCon India 2026
#
#  Targets:
#    make sim_axi_xsim      – AXI4 testbench (Vivado xsim)
#    make sim_core_xsim     – Core parallel scorer testbench (xsim)
#    make sim_axi_questa    – AXI4 testbench (Questa/ModelSim)
#    make sim_core_questa   – Core testbench (Questa)
#    make sim_axi_iverilog  – AXI4 testbench (Icarus Verilog)
#    make sim_core_iverilog – Core testbench (Icarus Verilog)
#    make clean             – Remove generated files
# =============================================================

RTL_DIR  := ../rtl
SIM_DIR  := .

# RTL source files (compile order: bottom-up)
RTL_SRCS := $(RTL_DIR)/MAC.v \
             $(RTL_DIR)/score.v \
             $(RTL_DIR)/obj_emb_bram.v \
             $(RTL_DIR)/scoring_lane.v \
             $(RTL_DIR)/parallel_scorer.v \
             $(RTL_DIR)/axi4_affinity_top.v

# Testbenches
AXI_TB   := $(RTL_DIR)/axi4_tb.v
CORE_TB  := $(RTL_DIR)/parallel_scorer_tb.v

# Top-level module names
AXI_TOP  := axi4_tb
CORE_TOP := parallel_scorer_tb

# Hex files (need to be in CWD for $readmemh)
HEX_FILES := $(RTL_DIR)/task_emb.hex \
              $(RTL_DIR)/obj_emb_0.hex \
              $(RTL_DIR)/obj_emb_1.hex

# ─── Copy hex files to sim directory ───────────────────────
.PHONY: copy_hex
copy_hex:
	@for f in $(HEX_FILES); do cp -f $$f $(SIM_DIR)/; done

# ─── Vivado xsim flow ─────────────────────────────────────
XVLOG := xvlog
XELAB := xelab
XSIM  := xsim

.PHONY: sim_axi_xsim sim_core_xsim
sim_axi_xsim: copy_hex
	@echo "[xsim] Compiling AXI testbench..."
	$(XVLOG) -work xwork $(RTL_SRCS) $(AXI_TB)
	$(XELAB) -work xwork $(AXI_TOP) -debug all -s axi_snap
	$(XSIM) axi_snap -runall -R

sim_core_xsim: copy_hex
	@echo "[xsim] Compiling Core testbench..."
	$(XVLOG) -work xwork $(RTL_SRCS) $(CORE_TB)
	$(XELAB) -work xwork $(CORE_TOP) -debug all -s core_snap
	$(XSIM) core_snap -runall -R

# ─── Questa / ModelSim flow ────────────────────────────────
VLIB := vlib
VLOG := vlog
VSIM := vsim

.PHONY: sim_axi_questa sim_core_questa
sim_axi_questa: copy_hex
	@echo "[Questa] Compiling AXI testbench..."
	$(VLIB) work
	$(VLOG) -sv $(RTL_SRCS) $(AXI_TB)
	$(VSIM) -c $(AXI_TOP) -do "run -all; quit"

sim_core_questa: copy_hex
	@echo "[Questa] Compiling Core testbench..."
	$(VLIB) work
	$(VLOG) -sv $(RTL_SRCS) $(CORE_TB)
	$(VSIM) -c $(CORE_TOP) -do "run -all; quit"

# ─── Icarus Verilog flow ──────────────────────────────────
IVERILOG := iverilog
VVP      := vvp

.PHONY: sim_axi_iverilog sim_core_iverilog
sim_axi_iverilog: copy_hex
	@echo "[iverilog] Compiling AXI testbench..."
	$(IVERILOG) -g2012 -o axi_tb.vvp $(RTL_SRCS) $(AXI_TB)
	$(VVP) axi_tb.vvp

sim_core_iverilog: copy_hex
	@echo "[iverilog] Compiling Core testbench..."
	$(IVERILOG) -g2012 -o core_tb.vvp $(RTL_SRCS) $(CORE_TB)
	$(VVP) core_tb.vvp

# Backwards-compatible alias
.PHONY: sim_xsim
sim_xsim: sim_axi_xsim

# ─── Clean ─────────────────────────────────────────────────
.PHONY: clean
clean:
	rm -rf xsim.dir .Xil xvlog.* xelab.* xsim.* *.log *.pb *.jou
	rm -rf work transcript vsim.wlf
	rm -f *.vcd *.wdb *.vvp webtalk*
	rm -f task_emb.hex obj_emb_0.hex obj_emb_1.hex
	@echo "[clean] Done."
