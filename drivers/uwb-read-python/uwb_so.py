import ctypes


# so文件在当前目录下，如trilatertion.so文件在/usr/lib下，请修改对应路径
lib_so = ctypes.cdll.LoadLibrary('./trilateration.so')
result = 0

class UWBMsg(ctypes.Structure):
     _fields_ = [("x", ctypes.c_double),
                ("y", ctypes.c_double),
                ("z", ctypes.c_double)]


location = UWBMsg()
anchorArray = (UWBMsg*8)()
distanceArray = (ctypes.c_int*8)(-1)


anchorArray[0].x = 0
anchorArray[0].y = 0
anchorArray[0].z = 2

anchorArray[1].x = 0
anchorArray[1].y = 10
anchorArray[1].z = 2

anchorArray[2].x = 10
anchorArray[2].y = 10
anchorArray[2].z = 2

anchorArray[3].x = 10
anchorArray[3].y = 0
anchorArray[3].z = 2

anchorArray[4].x = 20
anchorArray[4].y = 0
anchorArray[4].z = 2

anchorArray[5].x = 20
anchorArray[5].y = 10
anchorArray[5].z = 2

anchorArray[6].x = 30
anchorArray[6].y = 0
anchorArray[6].z = 2

anchorArray[7].x = 30
anchorArray[7].y = 10
anchorArray[7].z = 2

# 无效的测距值一定给 -1，否则会用随机数进行运算
distanceArray[0] = 7433
distanceArray[1] = 3905
distanceArray[2] = 8078
distanceArray[3] = -1
distanceArray[4] = 18848
distanceArray[5] = -1
distanceArray[6] = -1
distanceArray[7] = -1

# result = so.GetLocation(ctypes.byref(location),ctypes.c_int(1), anchorArray, distanceArray)
result = lib_so.GetLocation(ctypes.byref(location), anchorArray, distanceArray)

print(location.x)
print(location.y)
print(location.z)

print(result)
