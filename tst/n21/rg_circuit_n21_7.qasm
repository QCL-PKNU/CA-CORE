OPENQASM 2.0;
include "qelib1.inc";
qreg q[21];
creg c[21];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(4.658910381486251) q[1];
sx q[1];
rz(4.295750395830121) q[1];
sx q[1];
rz(12.308427118662658) q[1];
rz(pi) q[1];
x q[1];
rz(0) q[1];
sx q[1];
rz(7.700431093029442) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[3],q[5];
rz(5.530239116376867) q[5];
cx q[3],q[5];
rz(-pi/2) q[3];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(5.250118911960672) q[6];
sx q[6];
rz(4.7659072197713) q[6];
sx q[6];
rz(11.258781551591293) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
sx q[7];
rz(-pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/4) q[8];
cx q[2],q[8];
cx q[2],q[3];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[3];
rz(-pi/2) q[3];
rz(pi) q[3];
rz(-3.450616083308426) q[3];
sx q[3];
rz(8.65542527410724) q[3];
sx q[3];
rz(12.875394044077805) q[3];
rz(-0.5500478028641962) q[3];
rz(pi/2) q[3];
rz(-pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(-pi/2) q[8];
rz(pi/2) q[8];
rz(pi/4) q[9];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/4) q[12];
cx q[12],q[6];
rz(-pi/4) q[6];
cx q[12],q[6];
rz(-pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/4) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[13],q[4];
cx q[4],q[13];
rz(1.4109993405642722) q[13];
sx q[13];
rz(8.60806235534257) q[13];
sx q[13];
rz(11.681371379978136) q[13];
rz(2.1041516252437855) q[13];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/4) q[4];
cx q[9],q[4];
cx q[13],q[9];
rz(-pi/4) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(-pi/2) q[4];
rz(4.76641307372663) q[4];
rz(-2.1041516252437855) q[9];
cx q[13],q[9];
rz(2.1041516252437855) q[9];
rz(0.34761937543127275) q[9];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(1.977971817012776) q[14];
cx q[10],q[14];
rz(-1.977971817012776) q[14];
cx q[10],q[14];
rz(pi/2) q[10];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(2.5081712928265993) q[14];
sx q[14];
rz(6.348208658936804) q[14];
rz(0.4583388678710441) q[14];
cx q[5],q[10];
cx q[10],q[5];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(-pi/4) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/4) q[15];
rz(pi) q[15];
rz(pi/4) q[15];
cx q[15],q[2];
rz(-pi/4) q[2];
cx q[15],q[2];
rz(-pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/4) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[16],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(2.9721086450207124) q[17];
cx q[18],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/4) q[18];
cx q[18],q[0];
rz(-pi/4) q[0];
cx q[18],q[0];
rz(pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(2.3531027368862207) q[0];
rz(pi/2) q[0];
sx q[18];
cx q[7],q[0];
rz(0) q[0];
sx q[0];
rz(1.4865137461788949) q[0];
sx q[0];
rz(3*pi) q[0];
rz(0) q[7];
sx q[7];
rz(1.4865137461788949) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[7],q[0];
rz(-pi/2) q[0];
rz(-2.3531027368862207) q[0];
rz(pi) q[0];
x q[0];
rz(0) q[0];
sx q[0];
rz(3.2675017932508834) q[0];
sx q[0];
rz(3*pi) q[0];
cx q[1],q[0];
rz(0) q[0];
sx q[0];
rz(3.015683513928703) q[0];
sx q[0];
rz(3*pi) q[0];
cx q[1],q[0];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(0.7511005545658849) q[7];
cx q[4],q[7];
rz(-4.76641307372663) q[7];
sx q[7];
rz(0.7667665619610604) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[4],q[7];
rz(pi) q[4];
x q[4];
rz(0) q[7];
sx q[7];
rz(5.516418745218526) q[7];
sx q[7];
rz(13.440090479930124) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(1.851615696412859) q[7];
cx q[13],q[7];
rz(-1.851615696412859) q[7];
cx q[13],q[7];
rz(-pi/2) q[13];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(0.7331605323705989) q[7];
cx q[8],q[18];
rz(3.010813612403916) q[18];
sx q[18];
rz(5.238009841785685) q[18];
sx q[18];
rz(12.97612534371079) q[18];
rz(4.497905064396942) q[18];
rz(pi/2) q[18];
cx q[15],q[18];
rz(0) q[15];
sx q[15];
rz(0.642392830902407) q[15];
sx q[15];
rz(3*pi) q[15];
rz(0) q[18];
sx q[18];
rz(0.642392830902407) q[18];
sx q[18];
rz(3*pi) q[18];
cx q[15],q[18];
rz(-pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(1.538232563312871) q[15];
rz(pi/2) q[15];
cx q[0],q[15];
rz(0) q[0];
sx q[0];
rz(3.0118529033582457) q[0];
sx q[0];
rz(3*pi) q[0];
rz(0) q[15];
sx q[15];
rz(3.0118529033582457) q[15];
sx q[15];
rz(3*pi) q[15];
cx q[0],q[15];
rz(-pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
x q[0];
rz(-pi/2) q[15];
rz(-1.538232563312871) q[15];
rz(0) q[15];
sx q[15];
rz(6.155502999467702) q[15];
sx q[15];
rz(3*pi) q[15];
rz(-pi/2) q[18];
rz(-4.497905064396942) q[18];
rz(1.623313242248092) q[18];
x q[8];
rz(-3.74391918877486) q[8];
rz(pi/2) q[8];
cx q[12],q[8];
rz(0) q[12];
sx q[12];
rz(3.5543879400362948) q[12];
sx q[12];
rz(3*pi) q[12];
rz(0) q[8];
sx q[8];
rz(2.7287973671432915) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[12],q[8];
rz(-pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(-pi/2) q[12];
cx q[12],q[18];
rz(-1.623313242248092) q[18];
cx q[12],q[18];
cx q[13],q[12];
cx q[12],q[13];
rz(pi/2) q[12];
sx q[12];
rz(3.9044368707332575) q[12];
sx q[12];
rz(5*pi/2) q[12];
rz(pi/4) q[12];
rz(0.04956337948119952) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[12],q[13];
rz(-pi/4) q[13];
cx q[12],q[13];
rz(-pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/4) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(-pi/2) q[8];
rz(3.74391918877486) q[8];
rz(-pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(7.246354530166412) q[8];
sx q[8];
rz(5*pi/2) q[8];
rz(4.078303062053921) q[8];
sx q[8];
rz(3.9746338877215033) q[8];
sx q[8];
rz(13.244842932168476) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[19],q[17];
rz(-2.9721086450207124) q[17];
cx q[19],q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[17],q[16];
rz(4.796356647798257) q[16];
cx q[17],q[16];
rz(2.837324941047484) q[16];
cx q[16],q[6];
cx q[17],q[5];
rz(pi/4) q[17];
rz(pi/2) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/4) q[19];
cx q[11],q[19];
cx q[14],q[11];
rz(-0.4583388678710441) q[11];
cx q[14],q[11];
rz(0.4583388678710441) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[11],q[10];
rz(5.945601378247632) q[10];
cx q[11],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(4.527137561134375) q[14];
sx q[14];
rz(5.797705714359015) q[14];
sx q[14];
rz(15.298811344371561) q[14];
rz(pi/2) q[14];
cx q[10],q[14];
cx q[14],q[10];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[14],q[3];
rz(0) q[14];
sx q[14];
rz(3.846443232983402) q[14];
sx q[14];
rz(3*pi) q[14];
rz(-pi/4) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(-pi/2) q[19];
rz(pi/4) q[19];
rz(pi) q[19];
x q[19];
rz(1.0618653553518567) q[19];
rz(pi) q[19];
x q[19];
rz(-pi/2) q[19];
rz(pi/2) q[19];
rz(0) q[3];
sx q[3];
rz(2.436742074196184) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[14],q[3];
rz(-pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(0.029626487968670845) q[14];
rz(pi) q[14];
x q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-pi/2) q[3];
rz(0.5500478028641962) q[3];
rz(-pi/4) q[3];
rz(-0.5824468748621996) q[3];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[17],q[5];
rz(-pi/4) q[5];
cx q[17],q[5];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
rz(0) q[17];
sx q[17];
rz(8.158166998448426) q[17];
sx q[17];
rz(3*pi) q[17];
rz(2.0843022785638987) q[17];
rz(pi/4) q[17];
cx q[17],q[8];
rz(pi/4) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[5],q[2];
rz(2.1272513532636435) q[2];
cx q[5],q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[11],q[2];
cx q[2],q[11];
rz(pi/2) q[11];
cx q[2],q[10];
cx q[11],q[10];
rz(4.081028375847991) q[10];
cx q[11],q[10];
x q[10];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/4) q[11];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[2],q[19];
cx q[19],q[2];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(1.3410372930142216) q[19];
rz(5.396179583468735) q[2];
rz(1.5469649980629527) q[2];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[4],q[5];
cx q[4],q[15];
rz(0) q[15];
sx q[15];
rz(0.1276823077118845) q[15];
sx q[15];
rz(3*pi) q[15];
cx q[4],q[15];
sx q[15];
rz(pi) q[15];
rz(0) q[15];
sx q[15];
rz(4.7218708438330115) q[15];
sx q[15];
rz(3*pi) q[15];
cx q[10],q[15];
rz(0) q[15];
sx q[15];
rz(1.5613144633465745) q[15];
sx q[15];
rz(3*pi) q[15];
cx q[10],q[15];
rz(4.3332468944917455) q[10];
id q[10];
rz(0.7670233938511841) q[10];
sx q[10];
rz(5.607804802568623) q[10];
sx q[10];
rz(10.588290242824366) q[10];
cx q[4],q[0];
cx q[0],q[4];
rz(2.781706810030281) q[0];
cx q[0],q[11];
rz(pi) q[0];
rz(-pi/4) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(-pi/2) q[11];
rz(3.6562485764522314) q[11];
rz(-pi/2) q[11];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(2.990977198497242) q[5];
rz(0) q[5];
sx q[5];
rz(8.339917337239921) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[5],q[19];
rz(-1.3410372930142216) q[19];
cx q[5],q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/4) q[5];
rz(-2.837324941047484) q[6];
cx q[16],q[6];
rz(0) q[16];
sx q[16];
rz(4.353966058674194) q[16];
sx q[16];
rz(3*pi) q[16];
rz(-pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(2.837324941047484) q[6];
id q[6];
rz(-1.5910636295418685) q[6];
sx q[6];
rz(7.7044056579392715) q[6];
sx q[6];
rz(11.015841590311247) q[6];
cx q[6],q[18];
cx q[18],q[6];
cx q[6],q[18];
rz(-0.9206807695574044) q[18];
sx q[18];
rz(4.781574064143492) q[18];
sx q[18];
rz(10.345458730326783) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(-pi/2) q[6];
rz(-pi/4) q[8];
cx q[17],q[8];
rz(pi/4) q[17];
rz(pi/2) q[17];
sx q[17];
rz(4.350723276439274) q[17];
sx q[17];
rz(5*pi/2) q[17];
rz(pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(1.0238848574553596) q[8];
rz(pi) q[20];
x q[20];
rz(-pi/4) q[20];
rz(1.2145148719298022) q[20];
sx q[20];
rz(7.0437420530762616) q[20];
sx q[20];
rz(12.364498087290785) q[20];
cx q[9],q[20];
rz(-0.34761937543127275) q[20];
cx q[9],q[20];
rz(0.34761937543127275) q[20];
rz(2.830029384684251) q[20];
rz(pi/2) q[20];
cx q[1],q[20];
rz(0) q[1];
sx q[1];
rz(0.9405240423927821) q[1];
sx q[1];
rz(3*pi) q[1];
rz(0) q[20];
sx q[20];
rz(0.9405240423927821) q[20];
sx q[20];
rz(3*pi) q[20];
cx q[1],q[20];
rz(-pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(-pi/2) q[1];
cx q[1],q[6];
rz(1.5191512252111812) q[1];
cx q[1],q[3];
rz(-pi/2) q[20];
rz(-2.830029384684251) q[20];
x q[20];
rz(4.144428372823725) q[20];
cx q[20],q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/4) q[20];
cx q[20],q[16];
rz(-pi/4) q[16];
cx q[20],q[16];
rz(pi/4) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
sx q[16];
rz(-pi/2) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
cx q[20],q[12];
rz(0.04680290273603958) q[12];
cx q[20],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[18],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi) q[18];
x q[18];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(-1.5191512252111812) q[3];
sx q[3];
rz(1.1547974831109538) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[1],q[3];
rz(5.207630968209676) q[1];
rz(-pi/4) q[1];
rz(0) q[3];
sx q[3];
rz(5.128387824068632) q[3];
sx q[3];
rz(11.526376060842761) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[15],q[3];
rz(-pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[15],q[3];
rz(0.5026988732819867) q[3];
cx q[15],q[3];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(-pi/2) q[15];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/4) q[3];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[4],q[6];
rz(0.9327555061066647) q[6];
cx q[4],q[6];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(1.8685207878699461) q[4];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
cx q[6],q[16];
rz(5.02669702180109) q[16];
rz(-2.338101754248228) q[16];
sx q[16];
rz(5.141238386135659) q[16];
sx q[16];
rz(11.762879715017608) q[16];
cx q[16],q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[16];
x q[6];
cx q[6],q[20];
rz(5.878479513427397) q[20];
cx q[6],q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[20],q[6];
rz(2.868038419683783) q[6];
cx q[20],q[6];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi) q[20];
rz(pi/2) q[20];
rz(pi) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(0.6440999715521317) q[6];
rz(-pi/2) q[9];
cx q[7],q[9];
rz(-0.7331605323705989) q[9];
cx q[7],q[9];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi) q[7];
x q[7];
rz(1.5230521438707016) q[7];
cx q[7],q[17];
rz(-1.5230521438707016) q[17];
cx q[7],q[17];
rz(1.5230521438707016) q[17];
rz(pi/4) q[17];
id q[7];
rz(4.183670420177095) q[7];
rz(0.32834100758531637) q[7];
rz(0.7331605323705989) q[9];
rz(0.6565955386643834) q[9];
cx q[8],q[9];
rz(-1.0238848574553596) q[9];
cx q[8],q[9];
rz(1.3811298139231631) q[8];
cx q[2],q[8];
rz(-1.5469649980629527) q[8];
sx q[8];
rz(0.8976806717239065) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[2],q[8];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[5],q[2];
rz(-pi/4) q[2];
cx q[5],q[2];
rz(pi/4) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[2],q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[3],q[11];
rz(-pi/4) q[11];
cx q[3],q[11];
rz(pi/4) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-pi/4) q[5];
cx q[2],q[5];
cx q[5],q[2];
cx q[2],q[5];
rz(pi) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
sx q[5];
rz(-2.339188731781689) q[5];
sx q[5];
rz(8.982985352129653) q[5];
sx q[5];
rz(11.763966692551069) q[5];
cx q[6],q[11];
rz(-0.6440999715521317) q[11];
cx q[6],q[11];
rz(0.6440999715521317) q[11];
rz(1.3822368849378484) q[11];
rz(pi/4) q[11];
sx q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(0) q[8];
sx q[8];
rz(5.38550463545568) q[8];
sx q[8];
rz(9.590613144909168) q[8];
rz(pi/4) q[8];
cx q[8],q[19];
rz(-pi/4) q[19];
cx q[8],q[19];
rz(pi/4) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(1.5155402585301925) q[19];
cx q[12],q[19];
rz(-1.5155402585301925) q[19];
cx q[12],q[19];
rz(pi) q[12];
x q[12];
rz(pi/2) q[12];
sx q[12];
rz(3.323117396477949) q[12];
sx q[12];
rz(5*pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(-pi/2) q[19];
cx q[10],q[19];
rz(2.2862623147280114) q[19];
rz(4.254710475916817) q[19];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[17],q[8];
rz(-pi/4) q[8];
cx q[17],q[8];
sx q[17];
cx q[17],q[3];
rz(0) q[17];
sx q[17];
rz(7.5852412527745825) q[17];
sx q[17];
rz(3*pi) q[17];
cx q[17],q[6];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[3],q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/4) q[3];
cx q[3],q[2];
rz(-pi/4) q[2];
cx q[3],q[2];
rz(pi/4) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(2.123498817628617) q[2];
sx q[2];
rz(4.490583973465174) q[2];
sx q[2];
rz(13.502871097727633) q[2];
rz(pi/2) q[2];
rz(-pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(5.548517723696118) q[6];
cx q[17],q[6];
rz(pi/2) q[17];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(1.0238848574553596) q[9];
rz(2.3807390192645124) q[9];
cx q[13],q[9];
rz(-2.3807390192645124) q[9];
cx q[13],q[9];
cx q[14],q[13];
cx q[13],q[14];
rz(-pi/2) q[13];
cx q[0],q[13];
rz(1.6865904008802814) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[13];
rz(-pi/2) q[13];
rz(-pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[4],q[9];
rz(-1.8685207878699461) q[9];
cx q[4],q[9];
rz(pi/4) q[4];
cx q[4],q[14];
rz(-pi/4) q[14];
cx q[4],q[14];
rz(pi/4) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(1.584916229854909) q[14];
sx q[14];
rz(6.856448274635769) q[14];
rz(pi/2) q[14];
cx q[14],q[16];
x q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[10],q[14];
rz(0) q[10];
sx q[10];
rz(6.913028044565536) q[10];
sx q[10];
rz(3*pi) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[10],q[17];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(0.903318350269589) q[16];
cx q[17],q[10];
rz(pi) q[10];
rz(-0.07309016349186309) q[10];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
cx q[19],q[16];
rz(-4.254710475916817) q[16];
sx q[16];
rz(0.008767452985423141) q[16];
sx q[16];
rz(3*pi) q[16];
cx q[19],q[16];
rz(0) q[16];
sx q[16];
rz(6.274417854194163) q[16];
sx q[16];
rz(12.776170086416608) q[16];
rz(1.1690775318936046) q[16];
rz(-1.2076406530210975) q[19];
rz(pi/2) q[19];
cx q[12],q[19];
rz(0) q[12];
sx q[12];
rz(5.878293377226406) q[12];
sx q[12];
rz(3*pi) q[12];
rz(0) q[19];
sx q[19];
rz(0.4048919299531799) q[19];
sx q[19];
rz(3*pi) q[19];
cx q[12],q[19];
rz(-pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(-0.554751936827256) q[12];
rz(-pi/2) q[19];
rz(1.2076406530210975) q[19];
rz(4.640133045000985) q[19];
sx q[19];
rz(7.496646219787395) q[19];
sx q[19];
rz(13.047259835296114) q[19];
rz(-1.514857074772134) q[4];
rz(pi/2) q[4];
cx q[18],q[4];
rz(0) q[18];
sx q[18];
rz(5.58272746181053) q[18];
sx q[18];
rz(3*pi) q[18];
rz(0) q[4];
sx q[4];
rz(0.7004578453690562) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[18],q[4];
rz(-pi/2) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
cx q[0],q[18];
cx q[18],q[0];
rz(2.606912795375654) q[0];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[0],q[18];
rz(-2.606912795375654) q[18];
cx q[0],q[18];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[11],q[0];
rz(-pi/4) q[0];
cx q[11],q[0];
rz(pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi/2) q[0];
rz(pi/2) q[0];
rz(3.232516608574944) q[0];
sx q[0];
rz(8.899880432773553) q[0];
sx q[0];
rz(14.03594295539542) q[0];
rz(2.606912795375654) q[18];
rz(0.4820951949930916) q[18];
rz(-pi/2) q[4];
rz(1.514857074772134) q[4];
rz(-5.965201088131445) q[4];
rz(pi/2) q[4];
cx q[13],q[4];
rz(0) q[13];
sx q[13];
rz(4.598004287862528) q[13];
sx q[13];
rz(3*pi) q[13];
rz(0) q[4];
sx q[4];
rz(1.6851810193170587) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[13],q[4];
rz(-pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(-pi/2) q[4];
rz(5.965201088131445) q[4];
cx q[4],q[13];
rz(2.692187784979987) q[13];
cx q[4],q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[13],q[2];
cx q[2],q[13];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(-pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[3],q[4];
rz(5.037960605329684) q[4];
cx q[3],q[4];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(4.32708835660964) q[3];
cx q[3],q[12];
rz(-4.32708835660964) q[12];
sx q[12];
rz(0.551534806939741) q[12];
sx q[12];
rz(3*pi) q[12];
cx q[3],q[12];
rz(0) q[12];
sx q[12];
rz(5.731650500239845) q[12];
sx q[12];
rz(14.306618254206276) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(0.33613948265803106) q[12];
rz(-pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(1.8685207878699461) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/4) q[9];
cx q[1],q[9];
rz(-pi/2) q[1];
rz(-pi/4) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(-pi/2) q[9];
cx q[9],q[8];
rz(5.189527182545353) q[8];
cx q[9],q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(-pi/2) q[8];
cx q[1],q[8];
sx q[1];
cx q[5],q[1];
cx q[1],q[5];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(2.4289895671991224) q[1];
cx q[18],q[1];
rz(-2.4289895671991224) q[1];
cx q[18],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(0) q[1];
sx q[1];
rz(7.325956627517527) q[1];
sx q[1];
rz(3*pi) q[1];
rz(0.9505748364332692) q[1];
rz(4.561792462271038) q[1];
cx q[1],q[12];
rz(-4.561792462271038) q[12];
sx q[12];
rz(0.982145299491755) q[12];
sx q[12];
rz(3*pi) q[12];
cx q[1],q[12];
rz(0) q[12];
sx q[12];
rz(5.301040007687831) q[12];
sx q[12];
rz(13.650430940382385) q[12];
rz(1.9817950938081454) q[12];
rz(0) q[12];
sx q[12];
rz(7.3381621194953555) q[12];
sx q[12];
rz(3*pi) q[12];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(0.4725446021129916) q[18];
cx q[4],q[18];
rz(-0.4725446021129916) q[18];
cx q[4],q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(4.368706172487125) q[18];
rz(-pi/2) q[18];
rz(-pi/2) q[4];
cx q[2],q[4];
rz(2.9858451565588298) q[2];
rz(pi/2) q[4];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(1.6942361345867596) q[8];
rz(3.677241137580162) q[9];
rz(5.046525334276641) q[9];
cx q[9],q[7];
rz(-5.046525334276641) q[7];
sx q[7];
rz(0.09382235920474402) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[9],q[7];
rz(0) q[7];
sx q[7];
rz(6.189362947974843) q[7];
sx q[7];
rz(14.142962287460705) q[7];
cx q[15],q[7];
rz(3.702446392865265) q[7];
cx q[15],q[7];
rz(pi/4) q[15];
cx q[15],q[20];
rz(-pi/4) q[20];
cx q[15],q[20];
rz(5.70665175402956) q[15];
rz(0) q[15];
sx q[15];
rz(6.718207490432796) q[15];
sx q[15];
rz(3*pi) q[15];
rz(pi/4) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(3.6453033227991973) q[20];
rz(pi) q[20];
x q[20];
cx q[15],q[20];
cx q[20],q[15];
rz(-2.4998514844485715) q[20];
rz(pi) q[7];
x q[7];
rz(pi/2) q[7];
cx q[5],q[7];
cx q[7],q[5];
rz(pi/4) q[5];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[5],q[7];
rz(-pi/4) q[7];
cx q[5],q[7];
rz(5.373569248818062) q[5];
sx q[5];
rz(5.060396207538833) q[5];
sx q[5];
rz(12.967111653552418) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/4) q[5];
cx q[1],q[5];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi/4) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-pi/2) q[5];
rz(pi/4) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(2.6019492421523887) q[7];
cx q[9],q[8];
rz(-1.6942361345867596) q[8];
cx q[9],q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-pi/4) q[14];
rz(0) q[14];
sx q[14];
rz(6.619703940625488) q[14];
sx q[14];
rz(3*pi) q[14];
rz(pi/2) q[14];
sx q[14];
rz(9.281378877267166) q[14];
sx q[14];
rz(5*pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(1.7873426335020173) q[14];
cx q[0],q[14];
rz(-1.7873426335020173) q[14];
cx q[0],q[14];
id q[0];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi) q[8];
x q[8];
cx q[19],q[8];
cx q[8],q[19];
cx q[19],q[8];
rz(pi/4) q[8];
cx q[8],q[3];
rz(-pi/4) q[3];
cx q[8],q[3];
rz(pi/4) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[3],q[18];
rz(pi/2) q[18];
rz(4.892517456012912) q[3];
rz(pi/2) q[3];
rz(pi/2) q[8];
sx q[8];
rz(4.861262956512202) q[8];
sx q[8];
rz(5*pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[9],q[16];
rz(-1.1690775318936046) q[16];
cx q[9],q[16];
cx q[16],q[6];
rz(4.047975255333956) q[6];
cx q[16],q[6];
cx q[13],q[16];
cx q[16],q[13];
cx q[13],q[16];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(0.09098455986592278) q[13];
cx q[15],q[13];
rz(-0.09098455986592278) q[13];
cx q[15],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/4) q[13];
rz(0.7519219769541496) q[16];
cx q[16],q[10];
rz(-0.7519219769541496) q[10];
sx q[10];
rz(0.21759292341689385) q[10];
sx q[10];
rz(3*pi) q[10];
cx q[16],q[10];
rz(0) q[10];
sx q[10];
rz(6.065592383762692) q[10];
sx q[10];
rz(10.249790101215392) q[10];
x q[10];
rz(pi/2) q[10];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(-pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[16],q[3];
rz(0) q[16];
sx q[16];
rz(2.8106722735522425) q[16];
sx q[16];
rz(3*pi) q[16];
rz(0) q[3];
sx q[3];
rz(2.8106722735522425) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[16],q[3];
rz(-pi/2) q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(-pi/2) q[3];
rz(-4.892517456012912) q[3];
rz(3.669751176295997) q[6];
sx q[6];
rz(3.590942685491788) q[6];
rz(1.3542606053649864) q[9];
cx q[11],q[9];
rz(-1.3542606053649864) q[9];
cx q[11],q[9];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[11],q[17];
cx q[17],q[11];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(1.0326695897293048) q[17];
cx q[19],q[17];
rz(-1.0326695897293048) q[17];
cx q[19],q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
x q[19];
rz(pi/2) q[19];
sx q[19];
rz(4.576827165461848) q[19];
sx q[19];
rz(5*pi/2) q[19];
cx q[2],q[17];
rz(-2.9858451565588298) q[17];
cx q[2],q[17];
rz(2.9858451565588298) q[17];
rz(-pi/4) q[17];
rz(pi/4) q[2];
cx q[2],q[8];
cx q[6],q[11];
rz(4.472175377353694) q[11];
rz(3.44974341205117) q[11];
cx q[11],q[20];
rz(-3.44974341205117) q[20];
sx q[20];
rz(2.369363168728154) q[20];
sx q[20];
rz(3*pi) q[20];
cx q[11],q[20];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[13],q[11];
rz(-pi/4) q[11];
cx q[13],q[11];
rz(pi/4) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(0) q[20];
sx q[20];
rz(3.9138221384514322) q[20];
sx q[20];
rz(15.37437285726912) q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
cx q[20],q[10];
cx q[10],q[20];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(1.700812276827241) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[5],q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(-pi/4) q[8];
cx q[2],q[8];
rz(pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(5.961171643101263) q[9];
cx q[7],q[9];
rz(-2.6019492421523887) q[9];
cx q[7],q[9];
rz(pi/4) q[7];
rz(pi/2) q[7];
rz(2.6019492421523887) q[9];
cx q[4],q[9];
rz(1.3864946038168637) q[9];
cx q[4],q[9];
cx q[4],q[18];
rz(3.8451144528772256) q[18];
cx q[4],q[18];
cx q[9],q[15];
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
measure q[19] -> c[19];
measure q[20] -> c[20];