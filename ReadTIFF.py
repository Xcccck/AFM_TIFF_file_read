# -*- coding: utf-8 -*-            
# @Author : Achen
# @Time : 2023/10/23 14:30


import math
import os
import warnings

import pandas as pd
import numpy as np


def loadBinaryDatafromTIFF(fname):
    # 找到文件大小，然后以二进制格式读取
    nLen = int(np.ceil(os.path.getsize(fname) / 2))
    with open(fname, 'rb') as f:
        q = np.fromfile(f, dtype='<i4', count=nLen)

    # 检查是否存在读取错误
    if q.dtype == np.dtype('O'):
        warnings.warn(f"在{fname}中发现读取错误：NA。")
        q[q == 'NA'] = 0

    return q


def tagReader(tiffPth):
    q = loadBinaryDatafromTIFF(tiffPth)

    if not is_TIFF(q):
        raise Exception("Unrecognized TIFF format.")
    if is_big_endian(q):
        raise Exception("Big Endian encoding not supported yet.")

    A = q[1]
    tiffTags = pd.DataFrame()
    while 1:
        newtiffTags = read_IFD(q, A)
        if newtiffTags.shape[0] == 0:
            break
        tiffTags = pd.concat([tiffTags, newtiffTags], axis=0, ignore_index=False)
        A = newtiffTags.shape[0] * 12 + A + 2

    # 将一些数字转换为字符串
    tiffTags['tagName'] = identifyTIFFtags(tiffTags['tag'])
    tiffTags['typeName'] = identifyTIFFtypes(tiffTags['type'])

    # 对于“count”>1的项目，读取相关字段
    tiffTags["valueStr"] = ""
    # 读取ASCII字符串
    mASCII = np.where(tiffTags['type'] == 2)[0]
    for m1 in mASCII:
        tiffTags.at[m1, 'valueStr'] = readTIFF_ASCII(q, tiffTags.at[m1, 'value'], tiffTags.at[m1, 'count'])

    # 读取长数组
    mLong = np.where((tiffTags['type'] == 4) & (tiffTags['count'] > 1))[0]
    for m1 in mLong:
        tiffTags.loc[m1, 'valueStr'] = readTIFF_Long(q, tiffTags.loc[m1, 'value'], tiffTags.loc[m1, 'count'])

    # 读取短数组
    mShort = np.where((tiffTags['type'] == 3) & (tiffTags['count'] > 1))[0]
    for m1 in mShort:
        tiffTags.at[m1, 'valueStr'] = readColor_map(q, tiffTags.at[m1, 'value'], tiffTags.at[m1, 'count'])

    # 读取未知信息
    mUnkn = np.where((tiffTags['type'] == 7) & (tiffTags['count'] > 1))[0]
    for m1 in mUnkn:
        tiffTags.at[m1, 'valueStr'] = readUnknown(q, tiffTags.at[m1, 'value'], tiffTags.at[m1, 'count'])
    return tiffTags


def is_little_endian(q):
    return get16bit(q, 0) == 73 * 256 + 73


def is_big_endian(q):
    return get16bit(q, 0) == 77 * 256 + 77


def is_TIFF(q):
    return (is_little_endian(q) or is_big_endian(q)) and get16bit(q, 2) == 42


def get16bit(q, num):
    n = q[num // 4]
    if num % 4 == 0:
        res = n % 65536
    else:
        res = n // 65536
        if res < 0:
            res += 65536
    return res


def get32bit(q, num):
    if (num % 4) == 0:
        n = q[num // 4]
    else:
        n = get16bit(q, num + 2) * (2 ** 16) + get16bit(q, num)
    return n


def read_DirEntry(q, X):
    tagID = get16bit(q, X)
    if tagID < 0:
        tagID += (2 ** 16)
    return pd.DataFrame({
        'tag': [tagID],
        'type': [get16bit(q, X + 2)],
        'count': [get32bit(q, X + 4)],
        'value': [get32bit(q, X + 8)]
    })


def read_IFD(q, X):
    numDirEntries = get16bit(q, X)
    dIDF = pd.DataFrame()
    if numDirEntries > 0:
        for i in range(1, numDirEntries + 1):
            dIDF = pd.concat([dIDF, read_DirEntry(q, X + 2 + (i - 1) * 12)], axis=0, ignore_index=True)
    return dIDF


def identifyTIFFtags(tag_ID):
    return np.select(
        [
            tag_ID == 256, tag_ID == 257, tag_ID == 258, tag_ID == 259,
            tag_ID == 262, tag_ID == 263, tag_ID == 264, tag_ID == 265,
            tag_ID == 273, tag_ID == 274, tag_ID == 277,
            tag_ID == 278, tag_ID == 279,
            tag_ID == 305, tag_ID == 306, tag_ID == 315,
            tag_ID == 320,
            tag_ID == 50432, tag_ID == 50433,
            tag_ID == 50434, tag_ID == 50435, tag_ID == 50436,
            tag_ID == 50437, tag_ID == 50438, tag_ID == 50439
        ],
        [
            "ImageWidth", "ImageLength", "BitsPerSample", "Compression",
            "PhotometricInterpretation", "Thresholding", "CellWidth", "CellLength",
            "StripOffsets", "Orientation", "SamplesPerPixel",
            "RowsPerStrip", "StripByteCounts",
            "Software", "DateTime", "Artist",
            "ColorMap",
            "ParkMagicNumber", "ParkVersion",
            "ParkAFMdata", "ParkAFMheader", "ParkComments",
            "ParkLineProfile", "ParkSpectroHeader", "ParkSpectroData"
        ],
        default=""
    )


def identifyTIFFtypes(tags_type):
    return np.select(
        [
            tags_type == 1, tags_type == 2, tags_type == 3, tags_type == 4,
            tags_type == 5, tags_type == 6, tags_type == 7, tags_type == 8,
            tags_type == 9, tags_type == 10, tags_type == 11, tags_type == 12
        ],
        [
            "Byte", "ASCII", "Short (16-bit)", "Long (32-bit)",
            "Rational", "SByte", "Undefined", "SShort",
            "SLong", "SRational", "Float", "Double"
        ],
        default=""
    )


def readTIFF_ASCII(q, X, length):
    strASCII = []
    if (X % 2) == 1:
        X = X - 1
    for i in range(2, (length // 2) + 1):
        w2 = get16bit(q, X + (i - 1) * 2)
        strASCII.append(w2 % 256)
        strASCII.append(w2 // 256)
    return bytes(strASCII[0:length]).decode('utf-8')


def readTIFF_Long(q, X, len_):
    str_ = []
    for i in range(1, len_ + 1):
        str_.append(str(get32bit(q, X + (i - 1) * 4)))
    return ','.join(str_)


def readColor_map(q, X, length):
    cm = []
    for i in range(1, length + 1):
        w2 = get16bit(q, X + (i - 1) * 2)
        cm.append(w2)
    return ','.join(str(x) for x in cm)


def get_strip(q, X, length):
    A1 = (X // 4) + 1
    A2 = A1 + math.ceil(length // 4)
    n = q[(A1 - 1):A2]
    # convert n -> n8, so from 32-bits into 8-bit pieces
    n1 = n % (2 ** 8)
    n2 = n % (2 ** 16)
    n3 = n % (2 ** 24)
    n4 = (n - n3) // (2 ** 24)
    n3 = (n3 - n2) // (2 ** 16)
    n2 = (n2 - n1) // (2 ** 8)
    n4[n4 < 0] += (2 ** 8)
    n1 = pd.DataFrame(n1)
    n2 = pd.DataFrame(n2)
    n3 = pd.DataFrame(n3)
    n4 = pd.DataFrame(n4)
    n8 = pd.concat([n1, n2, n3, n4], axis=1, ignore_index=True)
    n8 = np.ravel(n8)
    B1 = (X + 1) % 4
    if B1 == 0:
        B1 = 4
    B2 = B1 + length - 1
    return n8[B1 - 1:B2]


def readUnknown(q, X, length):
    str_byte = get_strip(q, X, length)
    return ','.join(str(x) for x in str_byte)


def as32Bit(v):
    return v[3] * (2 ** 24) + v[2] * (2 ** 16) + v[1] * (2 ** 8) + v[0]


def byte2double(v):
    if sum(v) == 0:
        return 0
    v = v[::-1]  # reverse vector: little endian
    v2b = v[1] % 16
    v2a = v[1] // 16
    dbl_sgn_digit = v[0] // 128
    dbl_exp = (v[0] - dbl_sgn_digit * 128) * 16 + v2a - 1023
    dbl_mantissa = ((((1 * 2 ** 4 + v2b) * 2 ** 8 + v[2]) * 2 ** 8 + v[3]) * 2 ** 8 + v[4]) * 2 ** 8 + v[5]
    return dbl_mantissa * (2 ** (dbl_exp - 36)) * (sign((dbl_sgn_digit - 0.5) * (-2)))


def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


def get_ParkAFM_header(afm_params):
    df = pd.DataFrame({
        'imageType': [int(as32Bit(afm_params[0:4]))],
        'sourceName': [bytes([int(i) for i in afm_params[4:68] if int(i) != 0]).decode('utf-8')],
        'imageMode': [bytes([int(i) for i in afm_params[68:84] if int(i) != 0]).decode('utf-8')],
        'dfLPFStrength': [byte2double(afm_params[84:92])],
        'bAutoFlatten': [int(as32Bit(afm_params[92:96]))],
        'bACTrack': [int(as32Bit(afm_params[96:100]))],
        'nWidth': [int(as32Bit(afm_params[100:104]))],
        'nHeight': [int(as32Bit(afm_params[104:108]))],
        'dfAngle': byte2double(afm_params[108:116]),
        'bSineScan': [int(as32Bit(afm_params[116:120]))],
        'dfOverScan': byte2double(afm_params[120:128]),
        'bFastScanDir': [int(as32Bit(afm_params[128:132]))],
        'nSlowScanDir': [int(as32Bit(afm_params[132:136]))],
        'bXYSwap': [int(as32Bit(afm_params[136:140]))],

        'dfXScanSizeum': byte2double(afm_params[140:148]),
        'dfYScanSizeum': byte2double(afm_params[148:156]),
        'dfXOffsetum': byte2double(afm_params[156:164]),
        'dfYOffsetum': byte2double(afm_params[164:172]),
        'dfScanRateHz': byte2double(afm_params[172:180]),
        'dfSetPoint': byte2double(afm_params[180:188]),
        'SetPointUnitW': [bytes([int(i) for i in afm_params[188:204] if int(i) != 0]).decode('utf-8')],

        'dfTipBiasV': byte2double(afm_params[204:212]),
        'dfSampleBiasV': byte2double(afm_params[212:220]),
        'dfDataGain': byte2double(afm_params[220:228]),
        'dfZScale': byte2double(afm_params[228:236]),
        'dfZOffset': byte2double(afm_params[236:244]),

        'UnitZ': [bytes([int(i) for i in afm_params[244:260] if int(i) != 0]).decode('utf-8')],

        'nDataMin': [int(as32Bit(afm_params[260:264]))],
        'nDataMax': [int(as32Bit(afm_params[264:268]))],
        'nDataAvg': [int(as32Bit(afm_params[268:272]))],
        'nCompression': [int(as32Bit(afm_params[272:276]))],
        'bLogScale': [int(as32Bit(afm_params[276:280]))],
        'bSquare': [int(as32Bit(afm_params[280:284]))],

        'dfZServoGain': byte2double(afm_params[284:292]),
        'dfZScannerRange': byte2double(afm_params[292:300]),
        'XYVoltageMode': [bytes([int(i) for i in afm_params[300:316] if int(i) != 0]).decode('utf-8')],
        'ZVoltageMode': [bytes([int(i) for i in afm_params[316:332] if int(i) != 0]).decode('utf-8')],
        'XYServoMode': [bytes([int(i) for i in afm_params[332:348] if int(i) != 0]).decode('utf-8')],

        # Data Type 0=16bitshort, 1= 32bit int, 2= 32bit float
        'nDataType': [int(as32Bit(afm_params[348:352]))],
        'bXPDDRegion': [int(as32Bit(afm_params[352:356]))],
        'bYPDDRegion': [int(as32Bit(afm_params[356:360]))],

        'dfNCMAmplitude': byte2double(afm_params[360:368]),
        'dfNCMFrequency': byte2double(afm_params[368:376]),
        'dfHeadRotationAngle': byte2double(afm_params[376:384]),
        'Cantilever': [bytes([int(i) for i in afm_params[384:400] if int(i) != 0]).decode('utf-8')],

        # Non Contact Mode Drive %, range= 0-100
        'dfNCMDrivePercent': byte2double(afm_params[400:408]),
        'dfIntensityFactor': byte2double(afm_params[408:416])
    })
    return df


def get_value(tiff_tags, tag_name):
    val = None
    tag_no = tiff_tags.index[tiff_tags['tagName'] == tag_name].tolist()
    if len(tag_no) == 1:
        if tiff_tags.loc[tag_no[0], 'count'] == 1:
            val = tiff_tags.loc[tag_no[0], 'value']
        else:
            val = tiff_tags.loc[tag_no[0], 'valueStr']
    return val


def is_palette_color_image(tiff_tags):
    return get_value(tiff_tags, 'PhotometricInterpretation') == 3


def loadBinaryAFMDatafromTIFF(tiffPth, dataStart, dataLen, dataType):
    if dataType != 2:
        print("数据类型不是32位浮点数。")
    if dataLen % 4 != 0:
        print("数据长度不是32位倍数。")
    with open(tiffPth, 'rb') as f:
        f.seek(dataStart)
        q = np.frombuffer(f.read(), dtype='<f4', count=dataLen // 4)
        # q1 = f.read(dataStart)
    return q


def units2nanometer(unitZ):
    power = 1e3
    if unitZ == 'um':
        power = 1e3
    elif unitZ == 'nm':
        power = 1
    elif unitZ == 'deg':
        power = 1
    elif unitZ == 'mm':
        power = 1e6
    else:
        print("Unknown UnitZ: ", unitZ)
    return power


if __name__ == '__main__':
    tiffPth = './A3-1/EJC-AFM101&A3-1_0001_Height.tiff'
    tiffTags = tagReader(tiffPth)

    afm_params = list(map(float, tiffTags.loc[14, 'valueStr'].split(',')))
    head = get_ParkAFM_header(afm_params)

    if not is_palette_color_image(tiffTags):
        raise ValueError("Not a palette color image.")
    if not get_value(tiffTags, 'BitsPerSample') == 8:
        raise ValueError("Not an 8-bit image.")

    dataStart = tiffTags[tiffTags['tag'] == 50434]['value'].item()
    dataLen = tiffTags[tiffTags['tag'] == 50434]['count'].item()

    df = loadBinaryAFMDatafromTIFF(tiffPth, dataStart, dataLen, head.loc[0, 'nDataType'])
    imWidth = get_value(tiffTags, 'ImageWidth')
    imHeight = get_value(tiffTags, 'ImageLength')

    dataZ = (df * head.loc[0, 'dfDataGain']) * units2nanometer(head.loc[0, 'UnitZ'])

    data = np.array(np.reshape(dataZ, (head.loc[0, 'nHeight'], head.loc[0, 'nWidth'])))
