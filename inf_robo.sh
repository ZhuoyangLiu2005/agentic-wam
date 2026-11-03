I="There is a metal tong, a coiled small pack of plastic bags, and a red bowl on the table. Use your right hand to pick up the tong, move the tong's end to the pack of plastic bags, close the tong to lift it up, transport it to the red bowl, open the tong to drop the pack of plastic bags in the bowl, and place the tong down on the table."
# a) what the model will see + camera image size check; nothing moves
python deploy/robot/run_sharpa_eval.py --instruction "$I" --preview-only
# b) full loop with the guard, nothing sent to the robot
python deploy/robot/run_sharpa_eval.py --instruction "$I" --dry-run
# c) real run, someone on the e-stop
python deploy/robot/run_sharpa_eval.py --instruction "$I"
