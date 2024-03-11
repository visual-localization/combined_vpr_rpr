# 10/03/2024:
### Organization of Data 
- Should be changed as current organization is complicated - details are TBD

### Coordinate system of translation and orientation
- The current translation system is in the coordinate system depicted below with reference to orientation. => Need to re-orientated to get absolute position

![image](https://github.com/visual-localization/combined_vpr_rpr/assets/74974626/d10303d2-0f5c-4580-80ed-cc7c65ca7df9)

### Mapfree Dataset coordinate system
- Quaternion and Translation is given in the camera world coordinate system.
- In other worlds, from a point in the camera coordinate system of an image, its absolute position can be calculated as:

$abs_{point} = (cam_{point} - t) * q^{-1}$