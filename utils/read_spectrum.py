'''
Read spectrum (simulation/measurement)
'''
import numpy as np
import struct
import time

# read simulated spectrum 
def read_simu(path):
    '''
    Read MCNP simulated spectrum
    Return 
    ---------------------------
    dict contains:
        - energy bins
        - counts
    '''
    result = {}
    f = open(path,encoding = "ISO-8859-1")
    data_in = f.readlines()
    data_out=[]
    for line in range(len(data_in)):
        data_out.append(data_in[line].split())  
    for line in range(len(data_out)):
        if data_out[line]==['cell', '1']:
            if data_out[line+1]==['energy']:
                headline = line+6
        try:
            if data_out[line][0]=='total':
                if data_out[line+1][0]=='1analysis':
                    lastline = line
        except IndexError:
            continue
    data_list = data_out[headline:lastline]
    data = np.asanyarray(data_list)
    data = data.astype(np.float)
    non_dupl=np.where(np.diff(data[:,0])!=0) # remove duplicate energy bins
    result['energy'] = (data[non_dupl,0]).squeeze()
    result['counts'] = (data[non_dupl,1]).squeeze()
    return result

def readCNF(filename):
    '''
    Read measured spectrum
    Return 
    ---------------------------
    dict contains:
        - counts
        - channel
        - start_time
        - real_time
        - live_time
        - energy
        - A
    '''
    def uint8_at(f,pos):
        f.seek(pos)
        return np.fromfile(f,dtype=np.dtype('<u1'),count=1)[0]
    def uint16_at(f,pos):
        f.seek(pos)
        return np.fromfile(f,dtype=np.dtype('<u2'),count=1)[0]
    def uint32_at(f,pos):
        f.seek(pos)
        return np.fromfile(f,dtype=np.dtype('<u4'),count=1)[0]
    def uint64_at(f,pos):
        f.seek(pos)
        return np.fromfile(f,dtype=np.dtype('<u8'),count=1)[0]
    def pdp11f_at(f,pos):
        """
            Convert PDP11 32bit floating point format to
            IEE 754 single precision (32bits)
            """
        f.seek(pos)
        # Read two int16 numbers
        tmp16 = np.fromfile(f,dtype=np.dtype('<u2'),count=2)
        # Swapp positions
        mypack =  struct.pack('HH',tmp16[1],tmp16[0])
        f = struct.unpack('f',mypack)[0]/4.0
        return f
    def time_at(f,pos):
        #Period of time stored in units of 0.1 us
        return ~(uint64_at(f,pos))*1e-7
    def datetime_at(f,pos):
        return uint64_at(f,pos) / 10000000 - 3506716800
    def string_at(f,pos,length):
        f.seek(pos)
        # In order to avoid characters with not utf8 encoding
        return f.read(length).decode('utf8').rstrip('\00').rstrip()
    def calibration(channel,k):
        # calibrate energy bins
        energy_bar = k[0]+k[1]*channel+k[2]*np.power(channel,2)+k[3]*np.power(channel,3)
        return energy_bar
    f = open(filename,'rb')
    i = 0
    while True:
        sec_header = 0x70 + i*0x30
        sec_id_header = uint32_at(f,sec_header) # Section id in header
        sec_loc = uint32_at(f,sec_header+0x0a)  # start of section
        i+= 1
        if sec_id_header==0x00: break #no section
        elif sec_id_header==0x00012000:
            offs_param = sec_loc
        else: continue
    dic ={}
    i = 0
    while True:
        # length = 0x30
        sec_header = 0x70 + i*0x30
        sec_id_header = uint32_at(f,sec_header) # Section id in header
        sec_loc = uint32_at(f,sec_header+0x0a)  # start of section
        i+= 1
        if sec_id_header==0x00: break #no section
        elif sec_id_header==0x00012000:
            # time
            offs_times = sec_loc + 0x30 + uint16_at(f,sec_loc + 0x24)
            start_time = datetime_at(f,offs_times + 0x01)
            # Convert to formated date and time
            start_time = time.strftime('%d-%m-%Y, %H:%M:%S',time.gmtime(start_time))
            real_time = round(time_at(f,offs_times + 0x09))
            live_time = round(time_at(f,offs_times + 0x11))
            offs_calib = sec_loc + 0x30 + uint16_at(f, sec_loc + 0x22)
            A = np.empty(4)
            A[0] = pdp11f_at(f, offs_calib + 0x44)
            A[1] = pdp11f_at(f, offs_calib + 0x48)
            A[2] = pdp11f_at(f, offs_calib + 0x4c)
            A[3] = pdp11f_at(f, offs_calib + 0x50)
            #print(A)
            detector = string_at(f,offs_calib + 0x108,0x10)
        elif sec_id_header==0x00012001:
            #            if string_at(f,sec_loc + 0x477, 0x1)=='Y':
            #                dic['sample_name'] = string_at(f,sec_loc + 0x477, 0x40)
            #            else:
            #                dic['sample_name'] = string_at(f,sec_loc + 0x0030, 0x40) # named in data base
            sample_code = string_at(f,sec_loc + 0x0070, 0x40) # information about location of sampling
            geometry= string_at(f,sec_loc + 0xd4, 0x10)
            quality = string_at(f,sec_loc + 0x036e+0x40, 0x40)
        elif sec_id_header==0x00012002:
            geometrys = string_at(f,sec_loc + 0x10c, 0x10)
            detectors = string_at(f,sec_loc + 0x10c-6, 0x4)
        elif sec_id_header==0x00012005:
            Nchannels = uint8_at(f, offs_param + 0x00ba) * 256
            f.seek(sec_loc + 0x200)
            counts = np.dtype('int64').type(np.fromfile(f,dtype='<i4',count=Nchannels))
        else: continue
    dic['counts'] = counts
    dic['channel'] = np.arange(1,Nchannels+1)
    dic['start_time'] = start_time
    dic['real_time'] = real_time
    dic['live_time'] = live_time
    dic['energy'] = calibration(np.arange(1,Nchannels+1),A)
    dic['A'] = A
    if detector =='':
        dic['detector'] = detectors
    else:
        dic['detector'] = detector[-3:]
    dic['sample_code'] = sample_code
    dic['quality'] = quality
    # geometry (some of  them can not be read in section 0x00012001)
    # Here exeptions read in efficiency section 0x00012002...
    if geometry=='':
        if geometrys[0:4] == '60/2 ':
            dic['geometry'] = '30'
        else:
            dic['geometry'] = geometrys[0:2]
    elif geometry=='60/2 ml':
        dic['geometry'] = '30'
    else:
        dic['geometry'] = geometry[0:2]
    return dic

                
