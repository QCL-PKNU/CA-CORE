OPENQASM 2.0;
include "qelib1.inc";
qreg q[19];
creg c[19];
rz(4.41703140689061) q[0];
rz(pi/2) q[0];
id q[2];
rz(pi) q[2];
x q[2];
rz(-pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(-pi/4) q[3];
rz(pi) q[3];
x q[3];
rz(pi/4) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(0) q[3];
sx q[3];
rz(8.108886616629402) q[3];
sx q[3];
rz(3*pi) q[3];
x q[3];
rz(-pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-pi/4) q[4];
x q[4];
rz(pi) q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(3.0250258095420026) q[4];
rz(2.709528763606836) q[4];
rz(3.420392217258025) q[5];
rz(-pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
id q[8];
rz(1.3198835950927905) q[8];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[9],q[10];
rz(3.166162460318368) q[10];
cx q[9],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(0.6206112666046004) q[9];
sx q[9];
rz(3.3298113383322394) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(0.830162602772691) q[9];
cx q[11],q[7];
cx q[7],q[11];
rz(pi/2) q[11];
rz(-2.3211979281691137) q[11];
rz(pi/2) q[11];
cx q[2],q[11];
rz(0) q[11];
sx q[11];
rz(1.6656439845135362) q[11];
sx q[11];
rz(3*pi) q[11];
rz(0) q[2];
sx q[2];
rz(4.6175413226660496) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[2],q[11];
rz(-pi/2) q[11];
rz(2.3211979281691137) q[11];
rz(4.521961251737491) q[11];
rz(1.2843122285989528) q[11];
rz(-pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(1.1014029113907988) q[2];
sx q[2];
rz(6.076326536331133) q[2];
sx q[2];
rz(11.145371334102657) q[2];
rz(1.074693246093428) q[7];
rz(3.348312968806592) q[7];
cx q[7],q[8];
rz(-3.348312968806592) q[8];
sx q[8];
rz(2.4711431198820017) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[7],q[8];
rz(0) q[8];
sx q[8];
rz(3.8120421872975845) q[8];
sx q[8];
rz(11.453207334483182) q[8];
rz(-pi/2) q[8];
rz(-pi/2) q[12];
rz(-pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[12],q[5];
rz(2.3637125835530544) q[5];
cx q[12],q[5];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(0) q[13];
sx q[13];
rz(8.922485274796081) q[13];
sx q[13];
rz(3*pi) q[13];
rz(-pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[14],q[0];
rz(0) q[0];
sx q[0];
rz(2.6372528147283916) q[0];
sx q[0];
rz(3*pi) q[0];
rz(0) q[14];
sx q[14];
rz(2.6372528147283916) q[14];
sx q[14];
rz(3*pi) q[14];
cx q[14],q[0];
rz(-pi/2) q[0];
rz(-4.41703140689061) q[0];
rz(1.1137605599220997) q[0];
cx q[0],q[13];
rz(-1.1137605599220997) q[13];
cx q[0],q[13];
cx q[0],q[8];
rz(2.4850934987578057) q[0];
rz(1.1137605599220997) q[13];
rz(0.06548331374659117) q[13];
sx q[13];
rz(5.1169158672478225) q[13];
sx q[13];
rz(11.642352766222608) q[13];
rz(0.44971415879295595) q[13];
rz(-pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/4) q[15];
cx q[15],q[6];
rz(-pi/4) q[6];
cx q[15],q[6];
rz(pi/2) q[15];
cx q[15],q[14];
x q[14];
rz(-0.8319058916952122) q[14];
sx q[14];
rz(5.632899964030631) q[14];
rz(2.884968505398425) q[14];
x q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/4) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[6],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(1.0606193038943386) q[10];
sx q[10];
rz(3.9736118866540036) q[10];
sx q[10];
rz(9.671824443289493) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(-pi/4) q[6];
cx q[7],q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/4) q[7];
cx q[7],q[15];
rz(-pi/4) q[15];
cx q[7],q[15];
rz(pi/4) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(3.6811877264898487) q[15];
rz(1.589938725693712) q[15];
rz(-0.6914432196985648) q[7];
cx q[0],q[7];
rz(-2.4850934987578057) q[7];
sx q[7];
rz(1.8087506448199144) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[0],q[7];
rz(0) q[7];
sx q[7];
rz(4.474434662359672) q[7];
sx q[7];
rz(12.60131467922575) q[7];
rz(1.222563033745689) q[7];
cx q[7],q[0];
rz(-1.222563033745689) q[0];
cx q[7],q[0];
rz(1.222563033745689) q[0];
rz(pi) q[0];
x q[0];
sx q[0];
rz(0) q[0];
sx q[0];
rz(4.263756233254336) q[0];
sx q[0];
rz(3*pi) q[0];
rz(-1.6536649139565416) q[0];
rz(0) q[16];
sx q[16];
rz(5.173817763434439) q[16];
sx q[16];
rz(3*pi) q[16];
cx q[1],q[16];
rz(0) q[16];
sx q[16];
rz(1.1093675437451473) q[16];
sx q[16];
rz(3*pi) q[16];
cx q[1],q[16];
rz(0) q[1];
sx q[1];
rz(3.620560586641882) q[1];
sx q[1];
rz(3*pi) q[1];
rz(2.3939057763025127) q[1];
cx q[1],q[5];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(-2.3939057763025127) q[5];
cx q[1],q[5];
rz(-pi/2) q[1];
rz(2.3939057763025127) q[5];
cx q[5],q[10];
rz(1.746564138402239) q[10];
cx q[5],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(0) q[10];
sx q[10];
rz(4.843463507410426) q[10];
sx q[10];
rz(3*pi) q[10];
rz(0.11124627842689964) q[10];
rz(pi) q[10];
x q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/4) q[5];
cx q[6],q[1];
rz(pi/2) q[1];
rz(0.6649739231787971) q[1];
rz(pi/2) q[1];
rz(-1.2158029379553512) q[17];
sx q[17];
rz(7.3508123961013645) q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
rz(-1.0625282758504933) q[17];
cx q[11],q[17];
rz(-1.2843122285989528) q[17];
sx q[17];
rz(1.6525171150180864) q[17];
sx q[17];
rz(3*pi) q[17];
cx q[11],q[17];
rz(1.1665669818977473) q[11];
cx q[11],q[2];
rz(0) q[17];
sx q[17];
rz(4.6306681921615) q[17];
sx q[17];
rz(11.771618465218825) q[17];
rz(pi/2) q[17];
cx q[17],q[14];
rz(2.8458585348171272) q[14];
sx q[14];
rz(5.066897319331538) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-1.1665669818977473) q[2];
cx q[11],q[2];
cx q[11],q[5];
rz(1.1665669818977473) q[2];
cx q[2],q[7];
rz(-pi/4) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[7],q[2];
rz(-pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(3.340160752747077) q[7];
sx q[7];
rz(7.59815417426549) q[7];
rz(-pi/4) q[7];
rz(-pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
x q[18];
cx q[18],q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/4) q[18];
cx q[18],q[16];
rz(-pi/4) q[16];
cx q[18],q[16];
rz(pi/4) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(0.5852011938634094) q[16];
cx q[16],q[12];
rz(-0.5852011938634094) q[12];
cx q[16],q[12];
rz(0.5852011938634094) q[12];
rz(-1.0471829700637036) q[12];
cx q[13],q[16];
cx q[15],q[12];
rz(-1.589938725693712) q[12];
sx q[12];
rz(1.394957905641491) q[12];
sx q[12];
rz(3*pi) q[12];
cx q[15],q[12];
rz(0) q[12];
sx q[12];
rz(4.888227401538096) q[12];
sx q[12];
rz(12.061899656526794) q[12];
rz(1.7671420205234383) q[12];
cx q[12],q[6];
rz(2.983768137118356) q[15];
sx q[15];
rz(4.962488574183667) q[15];
sx q[15];
rz(10.206954142027225) q[15];
rz(-0.44971415879295595) q[16];
cx q[13],q[16];
rz(2.679119914045953) q[13];
sx q[13];
rz(7.9907135841564605) q[13];
sx q[13];
rz(14.340397606409553) q[13];
rz(-0.251851248126159) q[13];
sx q[13];
rz(4.9120074330749794) q[13];
sx q[13];
rz(9.676629208895537) q[13];
rz(0) q[13];
sx q[13];
rz(5.177112080445567) q[13];
sx q[13];
rz(3*pi) q[13];
rz(0.44971415879295595) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[17],q[13];
rz(0) q[13];
sx q[13];
rz(1.1060732267340194) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[17],q[13];
rz(pi/4) q[13];
cx q[13],q[14];
rz(-pi/4) q[14];
cx q[13],q[14];
rz(-3.8324161819694043) q[13];
sx q[13];
rz(3.7803684629493275) q[13];
sx q[13];
rz(13.257194142738783) q[13];
rz(pi/4) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/4) q[14];
rz(3.9958928288457445) q[17];
sx q[17];
rz(9.34163273337017) q[17];
sx q[17];
rz(12.6095907097177) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[18],q[9];
rz(-1.7671420205234383) q[6];
cx q[12],q[6];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(1.7671420205234383) q[6];
rz(0.013278674123157641) q[6];
cx q[4],q[6];
rz(-2.709528763606836) q[6];
sx q[6];
rz(0.9940834478291665) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[4],q[6];
rz(pi/4) q[4];
rz(0) q[6];
sx q[6];
rz(5.28910185935042) q[6];
sx q[6];
rz(12.121028050253058) q[6];
rz(pi/4) q[6];
cx q[6],q[5];
rz(-pi/4) q[5];
cx q[6],q[5];
rz(pi/4) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/4) q[6];
rz(-0.830162602772691) q[9];
cx q[18],q[9];
rz(0) q[18];
sx q[18];
rz(4.150239804811072) q[18];
sx q[18];
rz(3*pi) q[18];
cx q[18],q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi) q[16];
x q[16];
cx q[11],q[16];
cx q[16],q[11];
cx q[11],q[16];
rz(1.1677525786359617) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[18];
cx q[12],q[18];
cx q[18],q[12];
rz(2.592178697707959) q[12];
rz(pi/2) q[12];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(-2.6016238006503043) q[18];
sx q[18];
rz(9.126880485267693) q[18];
sx q[18];
rz(12.026401761419685) q[18];
rz(4.656450024506356) q[18];
sx q[18];
rz(2.073551900047352) q[18];
cx q[18],q[11];
rz(1.0781118766741606) q[11];
cx q[18],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/4) q[11];
cx q[2],q[12];
rz(0) q[12];
sx q[12];
rz(2.77550516070644) q[12];
sx q[12];
rz(3*pi) q[12];
rz(0) q[2];
sx q[2];
rz(2.77550516070644) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[2],q[12];
rz(-pi/2) q[12];
rz(-2.592178697707959) q[12];
cx q[16],q[12];
rz(0.37307392672905765) q[12];
cx q[16],q[12];
x q[16];
rz(0.7494684805684154) q[16];
id q[16];
rz(-pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(0.9722195154122185) q[2];
sx q[2];
rz(8.609490423398487) q[2];
sx q[2];
rz(15.184771578338587) q[2];
rz(3.021354689804174) q[2];
rz(pi/2) q[2];
cx q[7],q[2];
rz(0) q[2];
sx q[2];
rz(0.01915244509014391) q[2];
sx q[2];
rz(3*pi) q[2];
rz(0) q[7];
sx q[7];
rz(0.01915244509014391) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[7],q[2];
rz(-pi/2) q[2];
rz(-3.021354689804174) q[2];
rz(1.474296866444772) q[2];
rz(-pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(-0.9795085779309247) q[7];
cx q[2],q[7];
rz(-1.474296866444772) q[7];
sx q[7];
rz(2.5214560830427875) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[2],q[7];
rz(pi/4) q[2];
rz(pi/2) q[2];
rz(0) q[7];
sx q[7];
rz(3.7617292241367988) q[7];
sx q[7];
rz(11.878583405145076) q[7];
rz(pi/2) q[7];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/4) q[9];
cx q[9],q[8];
rz(-pi/4) q[8];
cx q[9],q[8];
rz(pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(-pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[1];
rz(0) q[1];
sx q[1];
rz(2.2662037507037502) q[1];
sx q[1];
rz(3*pi) q[1];
rz(0) q[8];
sx q[8];
rz(2.2662037507037502) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[8],q[1];
rz(-pi/2) q[1];
rz(-0.6649739231787971) q[1];
rz(3.049152350391508) q[1];
cx q[1],q[15];
rz(-3.049152350391508) q[15];
cx q[1],q[15];
rz(0.49939338873712136) q[1];
rz(pi/2) q[1];
rz(3.049152350391508) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[3],q[1];
rz(0) q[1];
sx q[1];
rz(2.779250248431899) q[1];
sx q[1];
rz(3*pi) q[1];
rz(0) q[3];
sx q[3];
rz(2.779250248431899) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[3],q[1];
rz(-pi/2) q[1];
rz(-0.49939338873712136) q[1];
cx q[1],q[6];
rz(-pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(-pi/4) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(-pi/2) q[6];
cx q[12],q[6];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(1.4095908267570239) q[12];
cx q[6],q[18];
rz(1.4914432782765157) q[6];
rz(-pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/4) q[8];
cx q[8],q[15];
rz(-pi/4) q[15];
cx q[8],q[15];
rz(pi/4) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(1.911173365137559) q[15];
rz(3.690075341389142) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[13],q[15];
rz(2.0473460300832023) q[15];
cx q[13],q[15];
rz(0.5952626706722278) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[10],q[8];
rz(pi/4) q[10];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[10],q[8];
rz(-pi/4) q[8];
cx q[10],q[8];
rz(5.526258408889077) q[10];
rz(2.8959428910009626) q[10];
cx q[10],q[0];
rz(-2.8959428910009626) q[0];
sx q[0];
rz(0.10913460675494902) q[0];
sx q[0];
rz(3*pi) q[0];
cx q[10],q[0];
rz(0) q[0];
sx q[0];
rz(6.174050700424637) q[0];
sx q[0];
rz(13.974385765726883) q[0];
cx q[0],q[12];
rz(pi) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[10],q[15];
rz(-1.4095908267570239) q[12];
cx q[0],q[12];
id q[0];
rz(-3.4202647646146405) q[0];
sx q[0];
rz(4.221556614861578) q[0];
sx q[0];
rz(12.845042725384019) q[0];
rz(-0.33137113853196115) q[0];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[15],q[10];
rz(4.375636668051825) q[10];
rz(-pi/4) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/4) q[15];
rz(pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[14];
rz(-pi/4) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-pi/2) q[14];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[14],q[8];
sx q[14];
cx q[7],q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[14],q[2];
cx q[2],q[14];
rz(pi/4) q[14];
rz(-pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
x q[7];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[6],q[8];
rz(-1.4914432782765157) q[8];
cx q[6],q[8];
rz(0.3563864973351847) q[6];
sx q[6];
rz(2.7412770789227556) q[6];
cx q[6],q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/4) q[6];
cx q[6],q[2];
rz(-pi/4) q[2];
cx q[6],q[2];
rz(pi/4) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(-pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
sx q[6];
rz(1.4914432782765157) q[8];
sx q[8];
rz(1.6764919433466632) q[9];
sx q[9];
rz(4.616362234597247) q[9];
sx q[9];
rz(12.567341502500309) q[9];
rz(0.5134979063179183) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[4],q[9];
rz(-pi/4) q[9];
cx q[4],q[9];
rz(1.1855912773223025) q[4];
cx q[4],q[3];
rz(-1.1855912773223025) q[3];
cx q[4],q[3];
rz(1.1855912773223025) q[3];
rz(pi/4) q[3];
cx q[3],q[17];
rz(-pi/4) q[17];
cx q[3],q[17];
rz(pi/4) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[12],q[17];
rz(-1.1264118882290222) q[12];
rz(pi/2) q[12];
rz(-pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[17],q[12];
rz(0) q[12];
sx q[12];
rz(2.7433932413297217) q[12];
sx q[12];
rz(3*pi) q[12];
rz(0) q[17];
sx q[17];
rz(3.5397920658498645) q[17];
sx q[17];
rz(3*pi) q[17];
cx q[17],q[12];
rz(-pi/2) q[12];
rz(1.1264118882290222) q[12];
rz(6.191992100670756) q[12];
rz(pi/2) q[12];
cx q[12],q[6];
x q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-pi/2) q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
rz(2.7946413252777105) q[17];
rz(pi/4) q[3];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(0.6261623793477505) q[4];
cx q[1],q[4];
rz(-0.6261623793477505) q[4];
cx q[1],q[4];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[11],q[1];
rz(-pi/4) q[1];
cx q[11],q[1];
rz(pi/4) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[18],q[11];
cx q[11],q[18];
rz(2.680132053872525) q[11];
sx q[11];
rz(4.047662195176965) q[11];
sx q[11];
rz(14.038884021934773) q[11];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(2.115277248056137) q[4];
rz(1.3812123554965017) q[4];
cx q[4],q[1];
rz(-1.3812123554965017) q[1];
cx q[4],q[1];
rz(1.3812123554965017) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[1],q[13];
rz(0.7879216790310974) q[13];
cx q[1],q[13];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(0.04081896795889455) q[1];
rz(-pi/2) q[1];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(2.716941821632318) q[13];
rz(-3.9836243678596093) q[4];
sx q[4];
rz(4.412789995904929) q[4];
sx q[4];
rz(13.408402328628988) q[4];
id q[4];
rz(0.6204025926525331) q[4];
sx q[4];
rz(4.927001387876617) q[4];
sx q[4];
rz(12.42870506042846) q[4];
rz(4.3485023539125836) q[4];
sx q[6];
cx q[7],q[18];
cx q[18],q[7];
cx q[13],q[18];
rz(-2.716941821632318) q[18];
cx q[13],q[18];
rz(pi/2) q[13];
sx q[13];
rz(5.352617008135985) q[13];
sx q[13];
rz(5*pi/2) q[13];
rz(2.716941821632318) q[18];
rz(-pi/4) q[7];
rz(-pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[2];
rz(5.622489223346144) q[2];
cx q[7],q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(1.4010339269259207) q[7];
cx q[13],q[7];
rz(-1.4010339269259207) q[7];
cx q[13],q[7];
rz(1.3117354419421772) q[13];
x q[13];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/4) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[5],q[9];
cx q[9],q[5];
cx q[5],q[9];
rz(pi/2) q[5];
sx q[5];
rz(5.371391596862703) q[5];
sx q[5];
rz(5*pi/2) q[5];
rz(0.9413129797724675) q[5];
sx q[5];
rz(6.040966256049563) q[5];
sx q[5];
rz(13.784386087980891) q[5];
rz(pi/2) q[5];
rz(3.431862535959091) q[5];
sx q[5];
rz(7.4911301156091055) q[5];
sx q[5];
rz(14.867174322375412) q[5];
cx q[5],q[11];
cx q[11],q[5];
cx q[5],q[11];
rz(pi/4) q[11];
rz(0.9219944749553031) q[5];
cx q[5],q[18];
rz(-0.9219944749553031) q[18];
cx q[5],q[18];
rz(0.9219944749553031) q[18];
cx q[2],q[18];
rz(1.6064449936383323) q[18];
cx q[2],q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[2],q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/4) q[2];
cx q[2],q[18];
rz(-pi/4) q[18];
cx q[2],q[18];
rz(pi/4) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi) q[18];
x q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi) q[5];
x q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[3],q[9];
rz(-pi/4) q[9];
cx q[3],q[9];
rz(5.81846614388145) q[3];
sx q[3];
rz(6.28917321985654) q[3];
sx q[3];
rz(13.964269515499016) q[3];
cx q[3],q[15];
rz(-pi/4) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(-pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(0.5828501715373569) q[3];
cx q[3],q[0];
rz(-0.5828501715373569) q[0];
sx q[0];
rz(0.46441613767848233) q[0];
sx q[0];
rz(3*pi) q[0];
cx q[3],q[0];
rz(0) q[0];
sx q[0];
rz(5.818769169501104) q[0];
sx q[0];
rz(10.338999270838697) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
cx q[0],q[6];
x q[0];
rz(pi) q[0];
x q[0];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[11],q[3];
rz(-pi/4) q[3];
cx q[11],q[3];
rz(-1.821817949015375) q[11];
rz(pi/4) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[4],q[11];
rz(-4.3485023539125836) q[11];
sx q[11];
rz(2.3496805140027757) q[11];
sx q[11];
rz(3*pi) q[11];
cx q[4],q[11];
rz(0) q[11];
sx q[11];
rz(3.9335047931768106) q[11];
sx q[11];
rz(15.595098263697338) q[11];
sx q[11];
rz(2.2352438674478243) q[6];
cx q[6],q[4];
rz(-2.2352438674478243) q[4];
cx q[6],q[4];
rz(2.2352438674478243) q[4];
rz(2.4565733716524436) q[4];
sx q[4];
rz(5.114409242219111) q[4];
cx q[13],q[4];
cx q[4],q[13];
rz(pi) q[13];
x q[13];
rz(pi/4) q[13];
x q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[8],q[15];
rz(6.115927888641592) q[15];
cx q[8],q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[10],q[15];
rz(2.1468075567293257) q[15];
cx q[10],q[15];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(-0.6730645624758652) q[10];
rz(pi/2) q[10];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(-pi/4) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(1.831091498717536) q[15];
cx q[3],q[10];
rz(0) q[10];
sx q[10];
rz(0.7992315149224196) q[10];
sx q[10];
rz(3*pi) q[10];
rz(0) q[3];
sx q[3];
rz(5.483953792257166) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[3],q[10];
rz(-pi/2) q[10];
rz(0.6730645624758652) q[10];
rz(-pi/4) q[10];
sx q[10];
rz(-pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
cx q[3],q[15];
rz(-1.831091498717536) q[15];
cx q[3],q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(0.9485579545340745) q[3];
cx q[0],q[3];
rz(-0.9485579545340745) q[3];
cx q[0],q[3];
rz(pi/2) q[0];
cx q[18],q[0];
cx q[0],q[18];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(2.528320158408213) q[8];
rz(pi/4) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(-0.7402391985787395) q[9];
sx q[9];
rz(5.597224229930482) q[9];
sx q[9];
rz(10.16501715934812) q[9];
rz(0) q[9];
sx q[9];
rz(6.041731338470827) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[16],q[9];
rz(0) q[9];
sx q[9];
rz(0.2414539687087589) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[16],q[9];
sx q[16];
cx q[16],q[1];
rz(pi/2) q[1];
rz(-pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[1],q[16];
rz(3.0172159337605935) q[16];
cx q[1],q[16];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
cx q[1],q[5];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(1.541212272105506) q[16];
cx q[17],q[9];
rz(0.28837532752830275) q[5];
cx q[1],q[5];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(3.6898925796836997) q[5];
rz(-2.7946413252777105) q[9];
cx q[17],q[9];
cx q[8],q[17];
rz(-2.528320158408213) q[17];
cx q[8],q[17];
rz(2.528320158408213) q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
id q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/4) q[8];
cx q[8],q[12];
rz(-pi/4) q[12];
cx q[8],q[12];
rz(pi/4) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[12],q[16];
rz(-1.541212272105506) q[16];
cx q[12],q[16];
rz(pi/4) q[12];
cx q[12],q[15];
rz(-pi/4) q[15];
cx q[12],q[15];
rz(pi/4) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/4) q[15];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[1],q[16];
rz(3.4298830518694055) q[16];
cx q[1],q[16];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(6.2519027925003865) q[1];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(4.699460956212032) q[16];
sx q[16];
rz(6.127097462052203) q[16];
sx q[16];
rz(14.206912275340095) q[16];
rz(2.9729787090507718) q[16];
cx q[16],q[18];
rz(-2.9729787090507718) q[18];
cx q[16],q[18];
rz(2.9729787090507718) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(2.3910013016584104) q[18];
cx q[5],q[15];
rz(-pi/4) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(-pi/2) q[15];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[0],q[5];
rz(pi/4) q[0];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[0],q[5];
rz(-pi/4) q[5];
cx q[0],q[5];
sx q[0];
rz(pi/4) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[5],q[18];
rz(-2.3910013016584104) q[18];
cx q[5],q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[8],q[7];
rz(5.631298287110991) q[7];
cx q[8],q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(-pi/2) q[7];
rz(pi/2) q[8];
cx q[8],q[10];
rz(3.571252902670043) q[10];
sx q[10];
rz(5.6866986754323685) q[10];
sx q[10];
rz(14.816551197807883) q[10];
rz(0) q[10];
sx q[10];
rz(5.188631701819812) q[10];
sx q[10];
rz(3*pi) q[10];
x q[8];
cx q[8],q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[8];
sx q[8];
rz(8.799949009279) q[8];
sx q[8];
rz(5*pi/2) q[8];
rz(2.7946413252777105) q[9];
rz(-1.706725336616607) q[9];
rz(pi/2) q[9];
cx q[14],q[9];
rz(0) q[14];
sx q[14];
rz(5.822909841218522) q[14];
sx q[14];
rz(3*pi) q[14];
rz(0) q[9];
sx q[9];
rz(0.46027546596106506) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[14],q[9];
rz(-pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
cx q[14],q[11];
cx q[11],q[6];
x q[14];
cx q[14],q[2];
rz(2.268553866462156) q[2];
cx q[14],q[2];
rz(0.5567355356880967) q[14];
rz(0.009114775560925992) q[2];
cx q[1],q[2];
rz(-6.2519027925003865) q[2];
sx q[2];
rz(3.084149593973928) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[1],q[2];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(0) q[2];
sx q[2];
rz(3.1990357132056584) q[2];
sx q[2];
rz(15.66756597770884) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[6],q[11];
cx q[11],q[6];
rz(pi/4) q[11];
cx q[11],q[17];
rz(-pi/4) q[17];
cx q[11],q[17];
rz(pi/4) q[11];
cx q[11],q[1];
rz(-pi/4) q[1];
cx q[11],q[1];
rz(pi/4) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-0.2616299508189963) q[1];
cx q[11],q[4];
rz(pi/4) q[11];
rz(pi/4) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[17],q[10];
rz(0) q[10];
sx q[10];
rz(1.0945536053597738) q[10];
sx q[10];
rz(3*pi) q[10];
cx q[17],q[10];
rz(-3.289359774795172) q[10];
sx q[10];
rz(4.721726579539007) q[10];
sx q[10];
rz(12.714137735564552) q[10];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[11],q[4];
rz(-pi/4) q[4];
cx q[11],q[4];
rz(pi/4) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(1.288595928968257) q[6];
cx q[6],q[12];
rz(-1.288595928968257) q[12];
cx q[6],q[12];
rz(1.288595928968257) q[12];
rz(pi/2) q[12];
sx q[12];
rz(3.578995805379253) q[12];
sx q[12];
rz(5*pi/2) q[12];
rz(1.2851753927812848) q[12];
cx q[15],q[6];
cx q[16],q[12];
rz(-1.2851753927812848) q[12];
cx q[16],q[12];
cx q[6],q[15];
rz(-pi/2) q[15];
rz(2.4212486103211948) q[6];
cx q[6],q[1];
rz(-2.4212486103211948) q[1];
sx q[1];
rz(1.6484359309618621) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[6],q[1];
rz(0) q[1];
sx q[1];
rz(4.634749376217724) q[1];
sx q[1];
rz(12.10765652190957) q[1];
cx q[7],q[14];
rz(-0.5567355356880967) q[14];
cx q[7],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[7],q[14];
rz(2.3955896424427108) q[14];
cx q[7],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[14],q[2];
rz(pi/4) q[14];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[14],q[2];
rz(-pi/4) q[2];
cx q[14],q[2];
rz(pi/4) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/4) q[7];
rz(-pi/2) q[9];
rz(1.706725336616607) q[9];
rz(pi) q[9];
x q[9];
rz(pi) q[9];
rz(4.819571818445656) q[9];
rz(2.7557622754196607) q[9];
sx q[9];
rz(5.040247478676671) q[9];
sx q[9];
rz(14.522666392486368) q[9];
sx q[9];
cx q[3],q[9];
x q[3];
cx q[3],q[8];
cx q[8],q[3];
x q[9];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
measure q[7] -> c[7];
measure q[8] -> c[8];
measure q[9] -> c[9];
measure q[10] -> c[10];
measure q[11] -> c[11];
measure q[12] -> c[12];
measure q[13] -> c[13];
measure q[14] -> c[14];
measure q[15] -> c[15];
measure q[16] -> c[16];
measure q[17] -> c[17];
measure q[18] -> c[18];
