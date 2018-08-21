import sys;
import os;

DOC_STR='javadoc -d .\FullyConnectedNetworkDocumentation *.java -html5'

COMP_STR = 'javac -verbose -d . *.java'
RUN_STR = 'java FullyConnectedNetwork.Network'



print("Creating Documentation...")
os.system(DOC_STR)
print("Done")
print("Compiling...")
os.system(COMP_STR)
print("Done")
