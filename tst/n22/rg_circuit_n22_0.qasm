OPENQASM 2.0;
include "qelib1.inc";
qreg q[22];
creg c[22];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(-pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(0) q[5];
sx q[5];
rz(3.1619905238525314) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[2],q[5];
rz(0) q[5];
sx q[5];
rz(3.121194783327055) q[5];
sx q[5];
rz(3*pi) q[5];
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
rz(4.823314410077025) q[6];
rz(3.059813719173354) q[6];
x q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(-2.129293605537347) q[9];
cx q[6],q[9];
rz(-3.059813719173354) q[9];
sx q[9];
rz(2.9226434368650356) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[6],q[9];
x q[6];
rz(-0.025954044358584446) q[6];
rz(pi/2) q[6];
rz(0) q[9];
sx q[9];
rz(3.3605418703145506) q[9];
sx q[9];
rz(14.61388528548008) q[9];
rz(0.3366654380115401) q[9];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(2.3535771574105855) q[10];
cx q[7],q[10];
rz(-2.3535771574105855) q[10];
cx q[7],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[10],q[5];
rz(5.873811497002438) q[5];
cx q[10],q[5];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(-2.5250273523753726) q[11];
rz(pi/2) q[11];
cx q[3],q[11];
rz(0) q[11];
sx q[11];
rz(0.0975553241925291) q[11];
sx q[11];
rz(3*pi) q[11];
rz(0) q[3];
sx q[3];
rz(6.185629982987058) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[3],q[11];
rz(-pi/2) q[11];
rz(2.5250273523753726) q[11];
cx q[11],q[7];
rz(-pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
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
rz(pi/2) q[2];
rz(-pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(1.5318638778588662) q[7];
cx q[11],q[7];
rz(pi/2) q[11];
sx q[11];
rz(4.073641753096187) q[11];
sx q[11];
rz(5*pi/2) q[11];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(-pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[3],q[7];
rz(2.832456252379052) q[7];
cx q[3],q[7];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(0) q[12];
sx q[12];
rz(7.5122830080308205) q[12];
sx q[12];
rz(3*pi) q[12];
cx q[12],q[9];
rz(-0.3366654380115401) q[9];
cx q[12],q[9];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/4) q[9];
rz(4.4762507695657785) q[13];
sx q[13];
rz(5.390508759287449) q[13];
rz(-pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(-pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-1.2201857500163542) q[15];
rz(pi/2) q[15];
cx q[14],q[15];
rz(0) q[14];
sx q[14];
rz(5.973746031745511) q[14];
sx q[14];
rz(3*pi) q[14];
rz(0) q[15];
sx q[15];
rz(0.30943927543407534) q[15];
sx q[15];
rz(3*pi) q[15];
cx q[14],q[15];
rz(-pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(-1.6953783041158943) q[14];
rz(pi/2) q[14];
cx q[13],q[14];
rz(0) q[13];
sx q[13];
rz(4.423623399982695) q[13];
sx q[13];
rz(3*pi) q[13];
rz(0) q[14];
sx q[14];
rz(1.8595619071968916) q[14];
sx q[14];
rz(3*pi) q[14];
cx q[13],q[14];
rz(-pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
rz(2.680502309949064) q[13];
rz(1.2986921349332183) q[13];
rz(-pi/2) q[14];
rz(1.6953783041158943) q[14];
rz(-pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[14],q[6];
rz(0) q[14];
sx q[14];
rz(6.232603722595842) q[14];
sx q[14];
rz(3*pi) q[14];
rz(-pi/2) q[15];
rz(1.2201857500163542) q[15];
rz(0) q[6];
sx q[6];
rz(0.05058158458374473) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[14],q[6];
rz(-pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(-pi/2) q[6];
rz(0.025954044358584446) q[6];
rz(pi) q[6];
x q[6];
rz(0.16746171165843116) q[6];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[4],q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi) q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[4],q[5];
cx q[5],q[4];
rz(3.546904848087739) q[4];
sx q[4];
rz(5.406086302861438) q[4];
rz(4.182737426355578) q[4];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(4.118844750978919) q[5];
rz(0.6117502270701546) q[5];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(0.7021784863927735) q[17];
cx q[0],q[17];
rz(-0.7021784863927735) q[17];
cx q[0],q[17];
rz(2.7256584748120787) q[0];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[17],q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/4) q[17];
cx q[17],q[16];
rz(-pi/4) q[16];
cx q[17],q[16];
cx q[1],q[17];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[11],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
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
rz(-pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/4) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(-pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(0) q[18];
sx q[18];
rz(9.185946480368145) q[18];
sx q[18];
rz(3*pi) q[18];
rz(pi/2) q[18];
cx q[18],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
sx q[18];
cx q[14],q[18];
x q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-1.3162022078993856) q[18];
rz(pi/2) q[18];
cx q[1],q[18];
rz(0) q[1];
sx q[1];
rz(3.790753939505879) q[1];
sx q[1];
rz(3*pi) q[1];
rz(0) q[18];
sx q[18];
rz(2.4924313676737073) q[18];
sx q[18];
rz(3*pi) q[18];
cx q[1],q[18];
rz(-pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/4) q[1];
rz(-pi/2) q[18];
rz(1.3162022078993856) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(-pi/2) q[19];
x q[19];
rz(-pi/2) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[16],q[19];
rz(3.685008672969052) q[19];
cx q[16],q[19];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
rz(pi) q[19];
rz(pi/4) q[19];
rz(0) q[19];
sx q[19];
rz(3.767716276345735) q[19];
sx q[19];
rz(3*pi) q[19];
cx q[7],q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/4) q[7];
cx q[7],q[16];
rz(-pi/4) q[16];
cx q[7],q[16];
rz(pi/4) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
cx q[14],q[16];
cx q[16],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[3];
rz(3.425327563552813) q[3];
cx q[7],q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
cx q[21],q[20];
cx q[0],q[21];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(2.2443808374742953) q[20];
cx q[20],q[15];
rz(-2.2443808374742953) q[15];
cx q[20],q[15];
rz(2.2443808374742953) q[15];
rz(pi/4) q[15];
cx q[15],q[10];
rz(-pi/4) q[10];
cx q[15],q[10];
rz(pi/4) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[10],q[17];
rz(-pi/4) q[15];
rz(-1.2100335061625465) q[15];
rz(2.9542098104165766) q[17];
cx q[10],q[17];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(2.849344708995945) q[10];
rz(2.7285352990362273) q[10];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(-1.8143012097257025) q[17];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(-2.7256584748120787) q[21];
cx q[0],q[21];
rz(2.7256584748120787) q[21];
sx q[21];
cx q[2],q[21];
cx q[12],q[21];
x q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[21],q[12];
rz(1.5836648651803609) q[12];
cx q[11],q[12];
rz(-1.5836648651803609) q[12];
cx q[11],q[12];
cx q[11],q[19];
rz(5.203520187212099) q[12];
rz(pi/2) q[12];
rz(0) q[12];
sx q[12];
rz(7.569507645558832) q[12];
sx q[12];
rz(3*pi) q[12];
rz(0) q[12];
sx q[12];
rz(3.281425670947966) q[12];
sx q[12];
rz(3*pi) q[12];
rz(0) q[19];
sx q[19];
rz(2.5154690308338514) q[19];
sx q[19];
rz(3*pi) q[19];
cx q[11],q[19];
rz(0) q[11];
sx q[11];
rz(6.503410936945973) q[11];
sx q[11];
rz(3*pi) q[11];
rz(-4.261261806612073) q[11];
rz(pi/2) q[11];
rz(-0.0923145283574957) q[19];
rz(pi/2) q[19];
rz(-2.696818120114789) q[21];
cx q[10],q[21];
rz(-2.7285352990362273) q[21];
sx q[21];
rz(1.6831121459564533) q[21];
sx q[21];
rz(3*pi) q[21];
cx q[10],q[21];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(0) q[21];
sx q[21];
rz(4.600073161223133) q[21];
sx q[21];
rz(14.850131379920395) q[21];
rz(3.104019392393168) q[21];
cx q[4],q[17];
rz(-4.182737426355578) q[17];
sx q[17];
rz(0.6440686690161281) q[17];
sx q[17];
rz(3*pi) q[17];
cx q[4],q[17];
rz(0) q[17];
sx q[17];
rz(5.639116638163458) q[17];
sx q[17];
rz(15.421816596850661) q[17];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[7],q[17];
cx q[17],q[7];
cx q[7],q[17];
rz(0.5322404551335822) q[17];
sx q[17];
rz(7.937011637709317) q[17];
sx q[17];
rz(10.811409547642002) q[17];
rz(-pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[4],q[7];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(5.014351640645295) q[4];
sx q[4];
rz(5*pi/2) q[4];
rz(-pi/2) q[4];
cx q[8],q[0];
rz(1.338850827625811) q[0];
cx q[8],q[0];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/4) q[8];
cx q[0],q[8];
rz(3.590050227646399) q[0];
rz(1.6435385994810503) q[0];
cx q[0],q[15];
rz(-1.6435385994810503) q[15];
sx q[15];
rz(1.6927268407340943) q[15];
sx q[15];
rz(3*pi) q[15];
cx q[0],q[15];
cx q[0],q[1];
cx q[0],q[18];
rz(-pi/4) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi/2) q[1];
rz(-pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[1],q[19];
rz(0) q[1];
sx q[1];
rz(4.685879824421259) q[1];
sx q[1];
rz(3*pi) q[1];
rz(0) q[15];
sx q[15];
rz(4.590458466445492) q[15];
sx q[15];
rz(12.278350066412976) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(pi/4) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(0.44448496426432627) q[18];
cx q[0],q[18];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[18],q[11];
rz(0) q[11];
sx q[11];
rz(1.817382368581813) q[11];
sx q[11];
rz(3*pi) q[11];
rz(0) q[18];
sx q[18];
rz(4.465802938597774) q[18];
sx q[18];
rz(3*pi) q[18];
cx q[18],q[11];
rz(-pi/2) q[11];
rz(4.261261806612073) q[11];
id q[11];
rz(-pi/2) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
id q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(0) q[19];
sx q[19];
rz(1.5973054827583268) q[19];
sx q[19];
rz(3*pi) q[19];
cx q[1],q[19];
rz(-pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(-pi/2) q[1];
rz(pi/4) q[1];
rz(2.8999597500243546) q[1];
rz(-pi/2) q[19];
rz(0.0923145283574957) q[19];
rz(5.857948413622789) q[19];
rz(pi/2) q[19];
cx q[17],q[19];
rz(0) q[17];
sx q[17];
rz(2.342729488916918) q[17];
sx q[17];
rz(3*pi) q[17];
rz(0) q[19];
sx q[19];
rz(2.342729488916918) q[19];
sx q[19];
rz(3*pi) q[19];
cx q[17],q[19];
rz(-pi/2) q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
rz(-pi/2) q[19];
rz(-5.857948413622789) q[19];
rz(pi/2) q[19];
sx q[19];
rz(6.736651678706455) q[19];
sx q[19];
rz(5*pi/2) q[19];
rz(-pi/2) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
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
rz(-pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(-3.1269299009132787) q[7];
sx q[7];
rz(7.336784671631761) q[7];
sx q[7];
rz(12.551707861682658) q[7];
rz(-pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(-pi/2) q[8];
cx q[8],q[5];
rz(-0.6117502270701546) q[5];
cx q[8],q[5];
sx q[5];
cx q[3],q[5];
x q[3];
rz(2.4740093826912) q[5];
cx q[9],q[20];
rz(-pi/4) q[20];
cx q[9],q[20];
rz(pi/4) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/4) q[20];
cx q[20],q[2];
rz(-pi/4) q[2];
cx q[20],q[2];
rz(pi/4) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(-0.07794540659390159) q[2];
sx q[2];
rz(2.53065273457549) q[2];
rz(-pi/2) q[2];
rz(4.819030228273987) q[20];
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
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(0) q[20];
sx q[20];
rz(5.52008790842955) q[20];
sx q[20];
rz(3*pi) q[20];
rz(-4.122639494305038) q[20];
rz(pi/2) q[20];
cx q[0],q[20];
rz(0) q[0];
sx q[0];
rz(6.07295051192055) q[0];
sx q[0];
rz(3*pi) q[0];
rz(0) q[20];
sx q[20];
rz(0.2102347952590362) q[20];
sx q[20];
rz(3*pi) q[20];
cx q[0],q[20];
rz(-pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(-pi/2) q[20];
rz(4.122639494305038) q[20];
rz(pi) q[20];
cx q[8],q[2];
rz(pi/2) q[2];
cx q[2],q[3];
rz(3.1759864544434424) q[3];
cx q[2],q[3];
cx q[2],q[3];
cx q[17],q[2];
cx q[2],q[17];
cx q[17],q[2];
rz(-pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(-pi/2) q[2];
rz(0) q[2];
sx q[2];
rz(5.269359432805217) q[2];
sx q[2];
rz(3*pi) q[2];
rz(pi/2) q[2];
id q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[7],q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(0.5369248025433484) q[3];
sx q[3];
rz(6.480271200471531) q[3];
sx q[3];
rz(12.342930259162708) q[3];
rz(pi/4) q[3];
x q[7];
rz(pi/2) q[8];
cx q[8],q[14];
rz(4.185994069006666) q[14];
id q[14];
rz(0.7545806933163508) q[14];
cx q[1],q[14];
rz(-2.8999597500243546) q[14];
sx q[14];
rz(1.9668083694867649) q[14];
sx q[14];
rz(3*pi) q[14];
cx q[1],q[14];
rz(0.6317214581945716) q[1];
rz(0) q[14];
sx q[14];
rz(4.316376937692821) q[14];
sx q[14];
rz(11.570157017477383) q[14];
x q[8];
rz(0) q[8];
sx q[8];
rz(5.5646093222576) q[8];
sx q[8];
rz(3*pi) q[8];
x q[8];
rz(pi/4) q[8];
rz(2.037673058236621) q[9];
cx q[9],q[13];
rz(-2.037673058236621) q[13];
sx q[13];
rz(0.5932241189375662) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[9],q[13];
rz(0) q[13];
sx q[13];
rz(5.68996118824202) q[13];
sx q[13];
rz(10.163758884072783) q[13];
cx q[6],q[13];
rz(-0.16746171165843116) q[13];
cx q[6],q[13];
rz(0.16746171165843116) q[13];
rz(-1.6852432965143969) q[13];
cx q[21],q[13];
rz(-3.104019392393168) q[13];
sx q[13];
rz(0.41300434034778233) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[21],q[13];
rz(0) q[13];
sx q[13];
rz(5.8701809668318035) q[13];
sx q[13];
rz(14.214040649676944) q[13];
rz(-0.8141274643577322) q[13];
rz(0.41865469050312415) q[21];
cx q[5],q[13];
rz(-2.4740093826912) q[13];
sx q[13];
rz(0.0377293067208746) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[5],q[13];
rz(0) q[13];
sx q[13];
rz(6.245456000458711) q[13];
sx q[13];
rz(12.712914807818311) q[13];
rz(pi/2) q[13];
sx q[5];
cx q[13],q[5];
x q[13];
rz(-3.0434059544498897) q[13];
rz(pi/2) q[13];
cx q[15],q[13];
rz(0) q[13];
sx q[13];
rz(1.4397901566609435) q[13];
sx q[13];
rz(3*pi) q[13];
rz(0) q[15];
sx q[15];
rz(4.843395150518643) q[15];
sx q[15];
rz(3*pi) q[15];
cx q[15],q[13];
rz(-pi/2) q[13];
rz(3.0434059544498897) q[13];
rz(0) q[13];
sx q[13];
rz(5.5242201101733) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[11],q[13];
rz(0) q[13];
sx q[13];
rz(0.7589651970062858) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[11],q[13];
id q[11];
rz(-pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
sx q[13];
rz(-pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(-5.500619007061344) q[15];
rz(pi/2) q[15];
cx q[17],q[15];
rz(0) q[15];
sx q[15];
rz(1.8645203295716615) q[15];
sx q[15];
rz(3*pi) q[15];
rz(0) q[17];
sx q[17];
rz(4.418664977607925) q[17];
sx q[17];
rz(3*pi) q[17];
cx q[17],q[15];
rz(-pi/2) q[15];
rz(5.500619007061344) q[15];
rz(-pi/2) q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
x q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
rz(-pi/4) q[17];
rz(-pi/2) q[17];
rz(4.0316084349263015) q[5];
cx q[20],q[5];
rz(3.932056075394886) q[5];
cx q[20],q[5];
rz(pi/2) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
rz(-1.4632981185494383) q[20];
rz(pi/2) q[20];
rz(1.8255437614948082) q[5];
sx q[5];
rz(6.499277511810638) q[5];
sx q[5];
rz(9.589939363871649) q[5];
rz(pi) q[6];
cx q[6],q[21];
rz(-0.41865469050312415) q[21];
cx q[6],q[21];
rz(-pi/4) q[21];
cx q[21],q[12];
rz(0) q[12];
sx q[12];
rz(3.00175963623162) q[12];
sx q[12];
rz(3*pi) q[12];
cx q[21],q[12];
rz(pi/2) q[12];
rz(1.4625323591718957) q[12];
cx q[12],q[15];
rz(-1.4625323591718957) q[15];
cx q[12],q[15];
rz(-pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[12],q[20];
rz(0) q[12];
sx q[12];
rz(3.28071674307464) q[12];
sx q[12];
rz(3*pi) q[12];
rz(1.4625323591718957) q[15];
rz(pi/2) q[15];
x q[15];
rz(2.3880018124504474) q[15];
rz(0) q[20];
sx q[20];
rz(3.0024685641049462) q[20];
sx q[20];
rz(3*pi) q[20];
cx q[12],q[20];
rz(-pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(-pi/2) q[20];
rz(1.4632981185494383) q[20];
rz(-pi/2) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[19],q[21];
rz(5.2785757803007725) q[21];
cx q[19],q[21];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
rz(pi/4) q[19];
rz(4.352323369755105) q[19];
rz(pi/2) q[19];
cx q[11],q[19];
rz(0) q[11];
sx q[11];
rz(1.92363612110548) q[11];
sx q[11];
rz(3*pi) q[11];
rz(0) q[19];
sx q[19];
rz(1.92363612110548) q[19];
sx q[19];
rz(3*pi) q[19];
cx q[11],q[19];
rz(-pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(-pi/2) q[19];
rz(-4.352323369755105) q[19];
rz(0.33312974772468706) q[19];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
sx q[6];
cx q[8],q[21];
cx q[21],q[8];
cx q[8],q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
sx q[21];
rz(3.0607328262187794) q[21];
rz(3.681773558417102) q[21];
rz(-pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(2.0934756605678015) q[9];
rz(4.698758797816745) q[9];
cx q[9],q[16];
rz(2.9475209391630366) q[16];
cx q[9],q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
cx q[16],q[6];
x q[16];
rz(pi) q[16];
rz(6.115957394120302) q[16];
rz(1.5495779551299982) q[16];
rz(5.787615887807083) q[16];
rz(pi/2) q[16];
rz(2.73578922890065) q[6];
rz(-pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[8],q[16];
rz(0) q[16];
sx q[16];
rz(1.405908043683411) q[16];
sx q[16];
rz(3*pi) q[16];
rz(0) q[8];
sx q[8];
rz(1.405908043683411) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[8],q[16];
rz(-pi/2) q[16];
rz(-5.787615887807083) q[16];
rz(pi/4) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(-pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(4.459846058334699) q[8];
cx q[8],q[19];
rz(-4.459846058334699) q[19];
sx q[19];
rz(1.5821067189439204) q[19];
sx q[19];
rz(3*pi) q[19];
cx q[8],q[19];
rz(0) q[19];
sx q[19];
rz(4.701078588235665) q[19];
sx q[19];
rz(13.55149427137939) q[19];
rz(0) q[19];
sx q[19];
rz(4.6062882754982395) q[19];
sx q[19];
rz(3*pi) q[19];
rz(0.7854307779235921) q[8];
rz(0.6417533811608677) q[8];
sx q[9];
cx q[10],q[9];
cx q[0],q[9];
x q[10];
rz(0.041779647602219465) q[10];
sx q[10];
rz(6.900865207468271) q[10];
sx q[10];
rz(12.640617248658009) q[10];
cx q[10],q[18];
rz(5.777378333755087) q[18];
cx q[10],q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[9],q[0];
cx q[0],q[9];
cx q[0],q[4];
cx q[0],q[14];
cx q[14],q[0];
cx q[0],q[14];
rz(0.2710380714375706) q[0];
sx q[0];
rz(5.775547949052646) q[0];
sx q[0];
rz(10.594938117389226) q[0];
sx q[14];
cx q[2],q[14];
rz(-0.022483835722496126) q[14];
sx q[14];
rz(7.348779011110263) q[14];
rz(5.784479877552005) q[14];
rz(0.39038182476083816) q[14];
sx q[14];
rz(5.723164393137868) q[14];
sx q[14];
rz(9.03439613600854) q[14];
rz(2.6898659386083663) q[14];
x q[2];
rz(2.189564446405524) q[2];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[18],q[4];
rz(pi/4) q[18];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[18],q[4];
rz(-pi/4) q[4];
cx q[18],q[4];
rz(1.984370036432252) q[18];
cx q[18],q[7];
rz(pi/4) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[4],q[5];
rz(0.6026399729862689) q[4];
sx q[4];
rz(6.956307919130151) q[4];
sx q[4];
rz(8.82213798778311) q[4];
rz(-pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[11],q[5];
rz(pi/2) q[11];
rz(2.1934569054612663) q[11];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-1.984370036432252) q[7];
cx q[18],q[7];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(2.4413594566065795) q[18];
cx q[0],q[18];
rz(-2.4413594566065795) q[18];
cx q[0],q[18];
x q[0];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(8.86043658556047) q[18];
sx q[18];
rz(5*pi/2) q[18];
sx q[18];
rz(1.984370036432252) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(-3.815802549643114) q[9];
rz(pi/2) q[9];
cx q[6],q[9];
rz(0) q[6];
sx q[6];
rz(6.161783294759838) q[6];
sx q[6];
rz(3*pi) q[6];
rz(0) q[9];
sx q[9];
rz(0.12140201241974813) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[6],q[9];
rz(-pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
cx q[6],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[3],q[13];
rz(-pi/4) q[13];
cx q[3],q[13];
rz(pi/4) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(3.335478865733185) q[13];
cx q[13],q[2];
rz(-3.335478865733185) q[2];
sx q[2];
rz(0.19661078605941418) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[13],q[2];
cx q[13],q[5];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(0) q[2];
sx q[2];
rz(6.086574521120172) q[2];
sx q[2];
rz(10.57069238009704) q[2];
rz(2.5741699309986914) q[2];
cx q[21],q[2];
rz(-3.681773558417102) q[2];
sx q[2];
rz(2.913709507930117) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[21],q[2];
rz(0) q[2];
sx q[2];
rz(3.3694757992494693) q[2];
sx q[2];
rz(10.532381588187791) q[2];
rz(0) q[2];
sx q[2];
rz(3.5197374442357985) q[2];
sx q[2];
rz(3*pi) q[2];
rz(3.5544267368555) q[3];
sx q[3];
rz(5.295387963611585) q[3];
sx q[3];
rz(13.71161658327073) q[3];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(0) q[5];
sx q[5];
rz(3.449271837636298) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[0],q[5];
rz(0) q[5];
sx q[5];
rz(2.833913469543288) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[0],q[5];
rz(2.97546175146283) q[0];
rz(pi/2) q[5];
sx q[5];
rz(9.001299885389166) q[5];
sx q[5];
rz(5*pi/2) q[5];
rz(4.943026304949554) q[5];
x q[6];
rz(-0.44637050105649845) q[6];
sx q[6];
rz(2.995434761176995) q[6];
rz(1.573553123357295) q[6];
cx q[6],q[20];
rz(-1.573553123357295) q[20];
cx q[6],q[20];
rz(1.573553123357295) q[20];
cx q[20],q[3];
cx q[3],q[20];
cx q[20],q[3];
rz(pi/2) q[20];
cx q[20],q[18];
rz(pi) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
x q[20];
rz(0.7903054400725692) q[20];
rz(2.426365056614197) q[3];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[6],q[16];
rz(1.77462402345311) q[16];
cx q[6],q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/4) q[16];
rz(-pi/2) q[16];
cx q[3],q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(3.9635805338407253) q[3];
rz(pi/2) q[3];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(0.6566486906183958) q[6];
sx q[6];
rz(4.6708149454857635) q[6];
sx q[6];
rz(8.768129270150984) q[6];
rz(-pi/2) q[9];
rz(3.815802549643114) q[9];
rz(3.060584127189221) q[9];
cx q[9],q[10];
rz(-3.060584127189221) q[10];
cx q[9],q[10];
rz(3.060584127189221) q[10];
rz(1.1031295366682283) q[10];
cx q[10],q[1];
rz(-1.1031295366682283) q[1];
cx q[10],q[1];
rz(1.1031295366682283) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(0.6395475687817881) q[1];
rz(pi) q[10];
x q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[12],q[1];
rz(-0.6395475687817881) q[1];
cx q[12],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[1],q[17];
rz(pi/4) q[1];
cx q[1],q[10];
rz(-pi/4) q[10];
cx q[1],q[10];
rz(pi/4) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[12],q[19];
cx q[14],q[1];
rz(-2.6898659386083663) q[1];
cx q[14],q[1];
rz(2.6898659386083663) q[1];
rz(pi/4) q[1];
cx q[1],q[18];
rz(-3.7472740557752022) q[14];
sx q[14];
rz(4.10568047926491) q[14];
sx q[14];
rz(13.172052016544582) q[14];
rz(-pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[17];
rz(-pi/4) q[18];
cx q[1],q[18];
id q[1];
rz(pi/4) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(5.357735806331856) q[18];
rz(0) q[19];
sx q[19];
rz(1.6768970316813467) q[19];
sx q[19];
rz(3*pi) q[19];
cx q[12],q[19];
rz(0.407622147822288) q[12];
cx q[21],q[19];
cx q[19],q[21];
cx q[21],q[19];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(5.493045829435934) q[21];
rz(pi/2) q[21];
cx q[6],q[10];
rz(6.269094233355627) q[10];
cx q[6],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(-2.529791284471518) q[10];
sx q[10];
rz(3.8609192255680647) q[10];
sx q[10];
rz(11.954569245240897) q[10];
rz(3.3525935119519445) q[10];
sx q[10];
rz(4.742446863444711) q[10];
sx q[10];
rz(9.779950712477604) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(-pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[6],q[3];
rz(0) q[3];
sx q[3];
rz(2.2859306786315066) q[3];
sx q[3];
rz(3*pi) q[3];
rz(0) q[6];
sx q[6];
rz(2.2859306786315066) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[6],q[3];
rz(-pi/2) q[3];
rz(-3.9635805338407253) q[3];
rz(-pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/4) q[6];
cx q[8],q[12];
rz(-0.6417533811608677) q[12];
sx q[12];
rz(1.9191217540340868) q[12];
sx q[12];
rz(3*pi) q[12];
cx q[8],q[12];
rz(0) q[12];
sx q[12];
rz(4.364063553145499) q[12];
sx q[12];
rz(9.65890919410796) q[12];
rz(1.9801578860313482) q[12];
rz(pi/4) q[12];
rz(3.0067939236956835) q[12];
sx q[12];
rz(6.353582700139084) q[12];
sx q[12];
rz(11.982593285541391) q[12];
rz(pi/4) q[12];
cx q[12],q[10];
rz(-pi/4) q[10];
cx q[12],q[10];
rz(pi/4) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(0) q[10];
sx q[10];
rz(8.695148101359969) q[10];
sx q[10];
rz(3*pi) q[10];
rz(-pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(2.6874983176339127) q[8];
sx q[8];
rz(7.246384553573885) q[8];
sx q[8];
rz(10.529394438091996) q[8];
rz(-pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[21];
rz(0) q[21];
sx q[21];
rz(2.9938003231267922) q[21];
sx q[21];
rz(3*pi) q[21];
rz(0) q[8];
sx q[8];
rz(2.9938003231267922) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[8],q[21];
rz(-pi/2) q[21];
rz(-5.493045829435934) q[21];
rz(pi/2) q[21];
rz(-pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(1.844292852212265) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[7],q[9];
rz(3.127584223755871) q[9];
cx q[7],q[9];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(-0.8935346604708609) q[7];
cx q[15],q[7];
rz(-2.3880018124504474) q[7];
sx q[7];
rz(0.7521543795867744) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[15],q[7];
cx q[15],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/4) q[15];
cx q[15],q[13];
rz(-pi/4) q[13];
cx q[15],q[13];
cx q[0],q[15];
rz(pi/4) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(-2.97546175146283) q[15];
cx q[0],q[15];
rz(1.326004656922513) q[0];
sx q[0];
rz(7.415880770701275) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(2.97546175146283) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(0) q[7];
sx q[7];
rz(5.531030927592812) q[7];
sx q[7];
rz(12.706314433690688) q[7];
cx q[7],q[2];
rz(0) q[2];
sx q[2];
rz(2.7634478629437877) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[7],q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[13],q[2];
rz(pi/4) q[13];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[13],q[2];
rz(-pi/4) q[2];
cx q[13],q[2];
rz(0) q[13];
sx q[13];
rz(8.559713788367048) q[13];
sx q[13];
rz(3*pi) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
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
rz(2.708377862377215) q[2];
cx q[20],q[2];
rz(-2.708377862377215) q[2];
cx q[20],q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(0.22433839332291328) q[2];
cx q[18],q[2];
rz(-5.357735806331856) q[2];
sx q[2];
rz(2.107194415599981) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[18],q[2];
rz(3.473140083257761) q[18];
sx q[18];
rz(4.988541295933814) q[18];
sx q[18];
rz(13.382690067207008) q[18];
rz(3.6712199667293266) q[18];
rz(0) q[2];
sx q[2];
rz(4.175990891579605) q[2];
sx q[2];
rz(14.558175373778322) q[2];
rz(-0.20046427592963068) q[2];
sx q[2];
rz(4.373559986000875) q[2];
sx q[2];
rz(9.62524223669901) q[2];
rz(-pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(1.5441010044551717) q[20];
rz(pi/2) q[20];
cx q[14],q[20];
rz(0) q[14];
sx q[14];
rz(0.8377445156454271) q[14];
sx q[14];
rz(3*pi) q[14];
rz(0) q[20];
sx q[20];
rz(0.8377445156454271) q[20];
sx q[20];
rz(3*pi) q[20];
cx q[14],q[20];
rz(-pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(-pi/2) q[20];
rz(-1.5441010044551717) q[20];
rz(-pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
cx q[6],q[13];
rz(-pi/4) q[13];
cx q[6],q[13];
rz(pi/4) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
sx q[13];
rz(1.829412397420868) q[6];
rz(pi/2) q[6];
sx q[6];
rz(6.602889000958468) q[6];
sx q[6];
rz(5*pi/2) q[6];
id q[7];
cx q[7],q[16];
rz(5.369360655277218) q[16];
cx q[7],q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(2.9404402449817226) q[16];
rz(-pi/2) q[7];
cx q[3],q[7];
rz(pi/2) q[3];
cx q[3],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
x q[3];
rz(pi/4) q[3];
rz(pi/2) q[7];
id q[7];
rz(-pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(1.776053147594325) q[9];
rz(pi/2) q[9];
cx q[4],q[9];
rz(0) q[4];
sx q[4];
rz(2.8202961133111244) q[4];
sx q[4];
rz(3*pi) q[4];
rz(0) q[9];
sx q[9];
rz(2.8202961133111244) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[4],q[9];
rz(-pi/2) q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
cx q[11],q[4];
rz(-2.1934569054612663) q[4];
cx q[11],q[4];
rz(6.027400323213241) q[11];
sx q[11];
rz(8.965149230248464) q[11];
sx q[11];
rz(15.614738930286299) q[11];
x q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/4) q[11];
rz(2.1934569054612663) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[19],q[4];
rz(pi/4) q[19];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[19],q[4];
rz(-pi/4) q[4];
cx q[19],q[4];
rz(-4.835926334334422) q[19];
rz(pi/2) q[19];
rz(pi/4) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(0) q[4];
sx q[4];
rz(6.427933167474057) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[4],q[11];
rz(-pi/4) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(-pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(5.350025765484403) q[4];
rz(pi/2) q[4];
cx q[20],q[4];
rz(0) q[20];
sx q[20];
rz(0.7867004477853929) q[20];
sx q[20];
rz(3*pi) q[20];
rz(0) q[4];
sx q[4];
rz(0.7867004477853929) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[20],q[4];
rz(-pi/2) q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
rz(1.063138042213214) q[20];
rz(-pi/2) q[4];
rz(-5.350025765484403) q[4];
rz(-pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[4],q[12];
rz(3.246951437322933) q[12];
cx q[4],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(-pi/2) q[9];
rz(-1.776053147594325) q[9];
cx q[9],q[17];
rz(pi/4) q[17];
rz(-pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[17],q[19];
rz(0) q[17];
sx q[17];
rz(5.903601459375958) q[17];
sx q[17];
rz(3*pi) q[17];
rz(0) q[19];
sx q[19];
rz(0.37958384780362886) q[19];
sx q[19];
rz(3*pi) q[19];
cx q[17],q[19];
rz(-pi/2) q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
rz(-pi/2) q[19];
rz(4.835926334334422) q[19];
rz(pi/4) q[19];
cx q[19],q[0];
rz(-pi/4) q[0];
cx q[19],q[0];
rz(pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
sx q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[5],q[17];
rz(4.737522592204685) q[17];
cx q[5],q[17];
x q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[3],q[17];
rz(-pi/4) q[17];
cx q[3],q[17];
rz(pi/4) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[5];
cx q[5],q[14];
rz(0.9962939973792568) q[14];
rz(pi/2) q[14];
cx q[2],q[14];
rz(0) q[14];
sx q[14];
rz(1.0166079734029791) q[14];
sx q[14];
rz(3*pi) q[14];
rz(0) q[2];
sx q[2];
rz(1.0166079734029791) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[2],q[14];
rz(-pi/2) q[14];
rz(-0.9962939973792568) q[14];
rz(-pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
x q[5];
rz(5.666911599847921) q[9];
cx q[9],q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/4) q[9];
cx q[9],q[15];
rz(-pi/4) q[15];
cx q[9],q[15];
rz(pi/4) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
sx q[15];
cx q[21],q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[11],q[15];
rz(0.6464976247191856) q[15];
cx q[11],q[15];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[20],q[11];
rz(-1.063138042213214) q[11];
cx q[20],q[11];
rz(1.063138042213214) q[11];
x q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/4) q[21];
cx q[0],q[21];
rz(-1.7495812215463462) q[0];
rz(pi/2) q[0];
rz(-pi/4) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(-pi/2) q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
cx q[5],q[15];
cx q[15],q[5];
cx q[5],q[15];
cx q[7],q[0];
rz(0) q[0];
sx q[0];
rz(1.6605405881482467) q[0];
sx q[0];
rz(3*pi) q[0];
rz(0) q[7];
sx q[7];
rz(4.6226447190313396) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[7],q[0];
rz(-pi/2) q[0];
rz(1.7495812215463462) q[0];
rz(-pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(-0.29097669970008666) q[9];
cx q[16],q[9];
rz(-2.9404402449817226) q[9];
sx q[9];
rz(1.2899386836090891) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[16],q[9];
rz(0.03432204442888885) q[16];
cx q[16],q[8];
rz(-0.03432204442888885) q[8];
cx q[16],q[8];
sx q[16];
rz(0.03432204442888885) q[8];
cx q[8],q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
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
rz(0) q[9];
sx q[9];
rz(4.9932466235704975) q[9];
sx q[9];
rz(12.656194905451189) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/4) q[9];
cx q[1],q[9];
rz(3.1588796760487003) q[1];
sx q[1];
rz(6.32803928868741) q[1];
sx q[1];
rz(12.43495783457698) q[1];
rz(-pi/4) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(-pi/2) q[9];
rz(pi/4) q[9];
cx q[9],q[13];
rz(-pi/4) q[13];
cx q[9],q[13];
rz(pi/4) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
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
measure q[21] -> c[21];