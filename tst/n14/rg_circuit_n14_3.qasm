OPENQASM 2.0;
include "qelib1.inc";
qreg q[14];
creg c[14];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(0.873920571689276) q[3];
sx q[3];
rz(3.758937788040482) q[3];
sx q[3];
rz(8.550857389080104) q[3];
sx q[3];
rz(-pi/4) q[3];
rz(0.13459860348660677) q[3];
rz(pi/2) q[3];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(0.22774105910964065) q[4];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[2],q[5];
rz(3.5150160878072243) q[5];
cx q[2],q[5];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/4) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/4) q[7];
rz(2.8488564120670614) q[8];
cx q[8],q[0];
rz(-2.8488564120670614) q[0];
cx q[8],q[0];
rz(2.8488564120670614) q[0];
cx q[8],q[0];
rz(0.5880647660448453) q[0];
cx q[8],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(0.2959589893202808) q[8];
rz(pi/4) q[9];
cx q[9],q[1];
rz(-pi/4) q[1];
cx q[9],q[1];
rz(pi/4) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/4) q[1];
cx q[1],q[2];
rz(-pi/4) q[2];
cx q[1],q[2];
rz(pi/4) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(1.5792702182259153) q[2];
sx q[2];
rz(5.498990811657228) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
cx q[9],q[7];
rz(-pi/4) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(-pi/2) q[7];
rz(pi/2) q[9];
cx q[10],q[4];
rz(-0.22774105910964065) q[4];
cx q[10],q[4];
rz(1.0832523196057058) q[10];
cx q[10],q[5];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(-1.0832523196057058) q[5];
cx q[10],q[5];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(1.0832523196057058) q[5];
sx q[5];
cx q[7],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(4.061947882502597) q[10];
rz(pi/2) q[10];
rz(pi/2) q[7];
rz(pi/2) q[7];
cx q[9],q[5];
rz(0.13919737814234012) q[5];
sx q[5];
rz(4.067210855840796) q[5];
rz(-pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
x q[9];
rz(pi) q[9];
rz(-3.2820166011614815) q[9];
rz(pi/2) q[9];
cx q[5],q[9];
rz(0) q[5];
sx q[5];
rz(4.5536001239732276) q[5];
sx q[5];
rz(3*pi) q[5];
rz(0) q[9];
sx q[9];
rz(1.7295851832063582) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[5],q[9];
rz(-pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(-pi/4) q[5];
rz(-pi/2) q[5];
rz(-pi/2) q[9];
rz(3.2820166011614815) q[9];
sx q[9];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(1.9412105913892623) q[11];
cx q[6],q[11];
rz(-1.9412105913892623) q[11];
cx q[6],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(3.7603958735875715) q[6];
rz(2.155759773075759) q[12];
sx q[12];
rz(9.117506921937697) q[12];
sx q[12];
rz(9.620128051896522) q[12];
sx q[12];
cx q[11],q[12];
x q[11];
rz(1.4070884601785896) q[11];
cx q[11],q[1];
rz(-1.4070884601785896) q[1];
cx q[11],q[1];
rz(1.4070884601785896) q[1];
rz(-pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[1],q[3];
rz(0) q[1];
sx q[1];
rz(1.4535014084319682) q[1];
sx q[1];
rz(3*pi) q[1];
rz(pi/4) q[12];
cx q[12],q[0];
rz(-pi/4) q[0];
cx q[12],q[0];
rz(pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[12],q[11];
rz(0.8639570988388564) q[11];
cx q[12],q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[11],q[7];
x q[12];
x q[12];
rz(0) q[3];
sx q[3];
rz(1.4535014084319682) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[1],q[3];
rz(-pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(2.618208974949205) q[1];
sx q[1];
rz(7.235430946510031) q[1];
sx q[1];
rz(13.238583700495074) q[1];
rz(-pi/2) q[3];
rz(-0.13459860348660677) q[3];
rz(-2.5302266628700183) q[3];
sx q[3];
rz(5.36244343409864) q[3];
sx q[3];
rz(11.955004623639397) q[3];
cx q[7],q[11];
rz(0) q[11];
sx q[11];
rz(9.039585626542681) q[11];
sx q[11];
rz(3*pi) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(5.144522965958791) q[7];
rz(1.7953459659401279) q[7];
rz(3.280993199444219) q[13];
sx q[13];
rz(3.0916928413628266) q[13];
cx q[13],q[4];
cx q[4],q[13];
cx q[13],q[4];
cx q[13],q[8];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[6],q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/4) q[6];
cx q[6],q[4];
rz(-pi/4) q[4];
cx q[6],q[4];
rz(pi/4) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(-pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[6],q[10];
rz(0) q[10];
sx q[10];
rz(2.395802770607354) q[10];
sx q[10];
rz(3*pi) q[10];
rz(0) q[6];
sx q[6];
rz(2.395802770607354) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[6],q[10];
rz(-pi/2) q[10];
rz(-4.061947882502597) q[10];
rz(-pi/4) q[10];
rz(-pi/2) q[10];
rz(-pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(-0.2959589893202808) q[8];
cx q[13],q[8];
rz(pi/2) q[13];
cx q[4],q[13];
cx q[13],q[4];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[13],q[2];
cx q[2],q[13];
rz(4.431452183359828) q[13];
sx q[13];
rz(6.934377572298329) q[13];
sx q[13];
rz(15.11352346142526) q[13];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[2],q[3];
rz(6.090250611516275) q[2];
cx q[2],q[11];
rz(3.872540215578242) q[11];
cx q[2],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(-pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
x q[2];
rz(-pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(0) q[4];
sx q[4];
rz(4.4113953426255215) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[6],q[4];
rz(0) q[4];
sx q[4];
rz(1.8717899645540643) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[6],q[4];
cx q[4],q[1];
rz(2.1997021952668456) q[1];
cx q[4],q[1];
rz(-pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[3],q[1];
rz(2.5442805621932862) q[1];
cx q[3],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[13],q[4];
rz(pi/4) q[13];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[13],q[4];
rz(-pi/4) q[4];
cx q[13],q[4];
rz(-pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/4) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi) q[4];
x q[4];
rz(-2.292797844689337) q[4];
rz(pi) q[6];
rz(pi) q[6];
rz(5.285109590305438) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[0],q[8];
rz(pi/4) q[0];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[0],q[8];
rz(-pi/4) q[8];
cx q[0],q[8];
rz(0.5129008792697233) q[0];
rz(0) q[0];
sx q[0];
rz(6.477622974026431) q[0];
sx q[0];
rz(3*pi) q[0];
cx q[7],q[0];
rz(-1.7953459659401279) q[0];
cx q[7],q[0];
rz(1.7953459659401279) q[0];
rz(-0.9465830853638895) q[0];
sx q[0];
rz(2.951440815595264) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(3.2133515200806357) q[7];
sx q[7];
rz(9.248047234872795) q[7];
sx q[7];
rz(13.745112435184007) q[7];
rz(-pi/2) q[7];
rz(-pi/2) q[7];
id q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(0.13858601384056266) q[7];
rz(pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(0) q[8];
sx q[8];
rz(8.012897170225049) q[8];
sx q[8];
rz(3*pi) q[8];
rz(pi/2) q[8];
cx q[8],q[9];
x q[8];
cx q[8],q[10];
rz(1.1356468980556527) q[10];
cx q[8],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[5],q[10];
rz(3.2364436824715144) q[10];
cx q[5],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[10],q[5];
rz(1.8556379699729513) q[5];
cx q[10],q[5];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/4) q[5];
cx q[5],q[10];
rz(-pi/4) q[10];
cx q[5],q[10];
rz(pi/4) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[10],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi) q[11];
x q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(0.991210312429776) q[5];
rz(pi) q[5];
x q[5];
rz(-pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(2.029965686019876) q[8];
cx q[8],q[1];
rz(-2.029965686019876) q[1];
cx q[8],q[1];
rz(2.029965686019876) q[1];
rz(4.142561732346501) q[1];
sx q[1];
rz(7.220680747344619) q[1];
sx q[1];
rz(13.67565300184544) q[1];
id q[1];
rz(pi) q[1];
x q[1];
x q[8];
rz(0) q[8];
sx q[8];
rz(3.8831548800435325) q[8];
sx q[8];
rz(3*pi) q[8];
rz(-pi/2) q[8];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[12],q[9];
rz(-pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[13],q[12];
rz(3.438364807236812) q[12];
cx q[13],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[12],q[0];
rz(1.7183553006213914) q[0];
cx q[12],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(0.27740840700881647) q[0];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[12],q[0];
rz(-0.27740840700881647) q[0];
cx q[12],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(2.7102421954093114) q[0];
cx q[0],q[2];
id q[12];
rz(-pi/4) q[12];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
cx q[13],q[6];
rz(pi/4) q[13];
rz(-2.7102421954093114) q[2];
cx q[0],q[2];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(2.7102421954093114) q[2];
rz(pi) q[2];
x q[2];
rz(4.367902006973884) q[2];
sx q[2];
rz(4.740668207683707) q[2];
rz(3.2667142886630924) q[2];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[13],q[6];
rz(-pi/4) q[6];
cx q[13],q[6];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/4) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[6],q[13];
rz(2.073159918766739) q[13];
cx q[6],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[13],q[8];
id q[13];
rz(0) q[13];
sx q[13];
rz(8.010947322424558) q[13];
sx q[13];
rz(3*pi) q[13];
rz(-pi/2) q[13];
rz(2.9936199740134084) q[13];
rz(-1.5075903268285202) q[13];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(2.4964953224107616) q[6];
sx q[6];
rz(9.263965335667137) q[6];
sx q[6];
rz(14.531093168376094) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[10],q[6];
rz(pi/4) q[10];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[10],q[6];
rz(-pi/4) q[6];
cx q[10],q[6];
rz(4.468428543859119) q[10];
sx q[10];
rz(4.737987743870169) q[10];
rz(pi/4) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(0.9772615343754294) q[6];
sx q[6];
rz(2.705302618596283) q[6];
rz(pi/2) q[8];
cx q[8],q[7];
rz(-0.13858601384056266) q[7];
cx q[8],q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(-pi/2) q[7];
rz(pi/4) q[8];
cx q[8],q[11];
rz(-pi/4) q[11];
cx q[8],q[11];
rz(pi/4) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
x q[11];
rz(1.464744213554183) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(2.643924358660174) q[9];
cx q[3],q[9];
rz(-2.643924358660174) q[9];
cx q[3],q[9];
rz(0.2380796541112713) q[3];
rz(2.5631489529011042) q[3];
cx q[3],q[4];
rz(-2.5631489529011042) q[4];
sx q[4];
rz(2.577644197174624) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[3],q[4];
rz(pi) q[3];
rz(-pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(0) q[4];
sx q[4];
rz(3.7055411100049622) q[4];
sx q[4];
rz(14.28072475835982) q[4];
rz(2.555512967172102) q[4];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
cx q[9],q[4];
rz(-2.555512967172102) q[4];
cx q[9],q[4];
rz(-pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[3],q[4];
rz(4.766004860674481) q[4];
cx q[3],q[4];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
cx q[3],q[0];
rz(1.4407369421094847) q[0];
cx q[3],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(0) q[0];
sx q[0];
rz(6.806013287826523) q[0];
sx q[0];
rz(3*pi) q[0];
rz(0.7679269298965212) q[3];
sx q[3];
rz(3.918214785671626) q[3];
id q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(pi) q[4];
x q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[12],q[4];
rz(6.235861144294072) q[4];
cx q[12],q[4];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(0.7718953224779975) q[12];
cx q[10],q[12];
rz(-0.7718953224779975) q[12];
cx q[10],q[12];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi) q[12];
x q[12];
rz(1.658407916166746) q[12];
sx q[12];
rz(4.8013458085598275) q[12];
sx q[12];
rz(9.946816427575017) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[3],q[10];
rz(1.8370365467219982) q[10];
cx q[3],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(0.6317277060663056) q[10];
sx q[10];
rz(4.928707079458669) q[10];
sx q[10];
rz(15.579398295136837) q[10];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(2.455363936172) q[3];
sx q[3];
rz(3.378919882502365) q[3];
sx q[3];
rz(13.12977072026673) q[3];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[8],q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/4) q[8];
cx q[8],q[4];
rz(-pi/4) q[4];
cx q[8],q[4];
rz(pi/4) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[4],q[6];
rz(-2.705302618596283) q[6];
cx q[4],q[6];
rz(pi/2) q[4];
rz(-pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(3.2761003390009096) q[6];
rz(5.898646662949144) q[8];
rz(3.306390447253876) q[9];
rz(2.754679860885856) q[9];
cx q[9],q[1];
rz(-2.754679860885856) q[1];
cx q[9],q[1];
rz(2.754679860885856) q[1];
rz(-pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[5],q[1];
rz(3.682209489666644) q[1];
cx q[5],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
cx q[1],q[0];
cx q[0],q[1];
x q[0];
rz(2.81215727010712) q[0];
rz(4.664055124767373) q[0];
rz(1.986348266677767) q[1];
cx q[2],q[1];
rz(-3.2667142886630924) q[1];
sx q[1];
rz(2.08276075157836) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[2],q[1];
rz(0) q[1];
sx q[1];
rz(4.200424555601226) q[1];
sx q[1];
rz(10.705143982754704) q[1];
rz(-1.8858267114024763) q[1];
rz(-0.7971022560927383) q[2];
cx q[0],q[2];
rz(-4.664055124767373) q[2];
sx q[2];
rz(0.1481379249450523) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[0],q[2];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(2.974455082299413) q[0];
cx q[10],q[0];
rz(-2.974455082299413) q[0];
cx q[10],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(1.672104554931734) q[0];
sx q[0];
rz(5.387345091000526) q[0];
sx q[0];
rz(15.28189725279088) q[0];
sx q[10];
sx q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(0) q[2];
sx q[2];
rz(6.135047382234534) q[2];
sx q[2];
rz(14.88593534162949) q[2];
rz(-pi/4) q[2];
rz(-pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(0) q[5];
sx q[5];
rz(5.063006728342897) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[6],q[1];
rz(-3.2761003390009096) q[1];
sx q[1];
rz(2.1343162296324207) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[6],q[1];
rz(0) q[1];
sx q[1];
rz(4.1488690775471655) q[1];
sx q[1];
rz(14.586705011172764) q[1];
x q[1];
rz(2.1158818332630425) q[1];
rz(-0.5319297438868973) q[6];
sx q[6];
rz(2.334826848628385) q[6];
rz(2.1912022765318904) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(0.986955995305594) q[6];
cx q[7],q[5];
rz(0) q[5];
sx q[5];
rz(1.2201785788366895) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[7],q[5];
rz(1.7200407531364528) q[5];
sx q[7];
cx q[7],q[8];
rz(3.0300525818360438) q[8];
cx q[7],q[8];
rz(-pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[4];
rz(5.503593234750545) q[4];
cx q[7],q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(-pi/2) q[4];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(-pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(-1.331584744070462) q[9];
sx q[9];
rz(5.712352023730158) q[9];
sx q[9];
rz(10.756362704839841) q[9];
rz(0) q[9];
sx q[9];
rz(8.03238844146983) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[5],q[9];
rz(-1.7200407531364528) q[9];
cx q[5],q[9];
rz(1.583667563516114) q[5];
cx q[5],q[13];
rz(-1.583667563516114) q[13];
sx q[13];
rz(1.7477359267798545) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[5],q[13];
rz(0) q[13];
sx q[13];
rz(4.535449380399732) q[13];
sx q[13];
rz(12.516035851114014) q[13];
rz(-pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[13],q[8];
rz(pi/2) q[5];
cx q[11],q[5];
cx q[5],q[11];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[5],q[7];
cx q[7],q[5];
rz(-pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(0.18876774122650955) q[7];
cx q[7],q[0];
rz(-0.18876774122650955) q[0];
cx q[7],q[0];
rz(0.18876774122650955) q[0];
cx q[0],q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(1.1384893861121759) q[7];
rz(1.6258148118173172) q[8];
cx q[13],q[8];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
cx q[13],q[12];
cx q[1],q[13];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(0.5274806270960597) q[13];
cx q[1],q[13];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/4) q[13];
cx q[13],q[10];
rz(-pi/4) q[10];
cx q[13],q[10];
rz(pi/4) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(3.8076011288270437) q[8];
x q[8];
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
rz(1.7200407531364528) q[9];
rz(pi/2) q[9];
cx q[9],q[3];
rz(1.004868526008049) q[3];
cx q[9],q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(2.7295127808702944) q[3];
cx q[11],q[3];
rz(-2.7295127808702944) q[3];
cx q[11],q[3];
rz(pi) q[11];
cx q[11],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[3],q[4];
cx q[4],q[3];
rz(pi) q[3];
x q[3];
rz(-pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[2];
rz(4.038347661131582) q[2];
cx q[9],q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
cx q[4],q[2];
cx q[2],q[4];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
cx q[9],q[6];
rz(-0.986955995305594) q[6];
cx q[9],q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[9],q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/4) q[9];
cx q[9],q[6];
rz(-pi/4) q[6];
cx q[9],q[6];
rz(pi/4) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
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
