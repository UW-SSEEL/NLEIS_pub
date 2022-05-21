This python code perform quick processing of NLEIS raw data and export the result to a save file.
After importing NLEIS.py, the function syntax is:
NLEIS(save_name, file_loc, freq_no, amp_no, galvano, IR, current_amplification, voltage_amplification)
Where:
	_ save_name: save location for the NLEIS data analysis results.
        _ file_loc: where the raw data located, with any prefixes.
        _ freq_no: number of studied frequency.
        _ amp_no: number of studied oscillation amplitude (alpha).
        _ galvano: True=galvanostatic experiment, False=potentiostatic experiment
        _ IR: potentiostat internal resistor for measuring current.
        _ current_amplification: amplification factor for current signal.
        _ voltage_amplification: amplification factor for voltage signal.