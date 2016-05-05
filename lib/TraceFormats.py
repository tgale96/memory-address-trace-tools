""" filename: TraceFormats
    contents: this file contains functions passed as callbacks to
    \"GenerateSyntheticTrace\" that are used to print the trace.
    all functiosn take in the arguments (traceFile, cycle, accessType, 
    memAddress), where traceFile is the handle to print the accesss to,
    cycle is an integer, accessType is a 0 or 1 for load and store
    respectively, and memAddress is a 4-byte word address
    
    author: Trevor Gale
    date: 3.5.16"""
    
def STL(traceFile, cycle, accessType, memAddress):
    """ STL: prints the memory reference in the format used in
        socket transaction language (STL)"""
    # handle read
    if not accessType:
        traceFile.write("%d: read 0x%x\n" % (cycle, memAddress))
    else:
        # default data for write is 0xABCD
        traceFile.write("%d: write 0x%x 0xABCD\n" % (cycle, memAddress))

def OVP(traceFile, cycle, accessType, memAddress):
    """ OVP: prints the memory reference in the format we used with 
        OVPsim for trace collection"""
    # handle read
    if not accessType:
        traceFile.write("r,0x%x\n" % (memAddress))
    else:
        traceFile.write("w,0x%x\n" % (memAddress))