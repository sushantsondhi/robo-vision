PY := python3

SRC1 := ./stereo.py
SRC2 := ./calibrate.py

INF1_1 := ./Input/left.jpg
INF1_2 := ./Input/right.jpg
INF1_3 := ./Input/part1.pkl
INF2_1 := ./Input/left1.16mm.pkl
INF2_2 := ./Input/left2.16mm.pkl
OUTF_1 := ./Output/out1.txt
ERRF_1 := ./Output/err1.txt
OUTF_2 := ./Output/out2.txt
ERRF_2 := ./Output/err2.txt

.PHONY: part1 part2 debug clean

.DEFAULT_GOAL := clean

part1: $(SRC1)
	@printf "Part1: Stereo\n";
	@$(PY) $(SRC1) $(INF1_1) $(INF1_2) $(INF1_3) 2> $(ERRF_1) > $(OUTF_1)

part2: $(SRC2)
	@printf "Part2: Camera Calibration\n";
	@$(PY) $(SRC2) $(INF2_1) $(INF2_2) 2> $(ERRF_2) > $(OUTF_2)

debug1: $(SRC1)
	@printf "Part1: Stereo\n";
	@$(PY) $(SRC1) $(INF1_1) $(INF1_2) $(INF1_3)

debug2: $(SRC2)
	@printf "Part2: Camera Calibration\n";
	@$(PY) $(SRC2) $(INF2_1) $(INF2s_2)

clean:
	@rm -rv $(OUTF_1) $(OUTF_2) $(ERRF_1) $(ERRF_2)
