"""Bench tooling for the Feetech HD-1910-C001 servo (HLS protocol): bus access, a simulated servo for
hardware-free runs, and the BAM actuator + fit used to replace the provisional sim model.

Numpy, scipy and bam only (no torch, no mjlab), so the scripts start fast on a laptop next to the rig.
See docs/bench.md for the session procedure.
"""
