OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
creg c[25];
rz(5.853629672251412) q[0];
sx q[0];
rz(8.01597374324378) q[0];
sx q[0];
rz(15.543883940578887) q[0];
rz(4.624483958549855) q[0];
rz(pi/2) q[0];
rz(pi/4) q[1];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(2.212911375054119) q[3];
rz(pi) q[4];
x q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[2],q[5];
cx q[5],q[2];
rz(2.375573599783322) q[5];
rz(pi/4) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(3.817695451557766) q[7];
sx q[7];
rz(2.035866862787869) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[4],q[7];
rz(1.2325929815040142) q[7];
cx q[4],q[7];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(0.13880343882691823) q[4];
rz(1.9295722396295032) q[4];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[1],q[8];
rz(-pi/4) q[8];
cx q[1],q[8];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(-4.031213307855425) q[8];
rz(pi/2) q[8];
id q[9];
rz(0.9120747143180084) q[9];
sx q[9];
rz(4.1348815119805975) q[9];
rz(4.665799900874706) q[9];
rz(2.758367277373509) q[9];
rz(2.085691081599869) q[10];
sx q[10];
rz(6.977387510964498) q[10];
sx q[10];
rz(15.29445761987786) q[10];
rz(pi/2) q[10];
rz(2.8567044892918547) q[11];
sx q[11];
rz(6.26480451799933) q[11];
cx q[11],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(4.5443370400239) q[1];
sx q[1];
rz(4.661507701289294) q[1];
sx q[1];
rz(9.624360535069835) q[1];
rz(pi/4) q[1];
rz(pi/2) q[11];
sx q[11];
rz(3.8001161272244275) q[11];
sx q[11];
rz(5*pi/2) q[11];
rz(pi/4) q[11];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[13],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(3.463006863780044) q[12];
sx q[12];
rz(7.219587414180446) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
id q[13];
rz(4.331735686078017) q[14];
sx q[14];
rz(7.629502117055533) q[14];
sx q[14];
rz(10.990767046311907) q[14];
rz(-1.4675281932118518) q[14];
cx q[5],q[14];
rz(-2.375573599783322) q[14];
sx q[14];
rz(2.3579025667769655) q[14];
sx q[14];
rz(3*pi) q[14];
cx q[5],q[14];
rz(0) q[14];
sx q[14];
rz(3.9252827404026207) q[14];
sx q[14];
rz(13.267879753764554) q[14];
rz(pi/2) q[5];
sx q[5];
rz(7.730973088179187) q[5];
sx q[5];
rz(5*pi/2) q[5];
rz(pi/2) q[5];
rz(pi/4) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(-0.4447697079029518) q[16];
sx q[16];
rz(8.495237705237333) q[16];
sx q[16];
rz(9.86954766867233) q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
cx q[16],q[12];
rz(1.4428099682344742) q[12];
cx q[16],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(6.514238694039522) q[12];
sx q[12];
rz(5*pi/2) q[12];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[17];
sx q[17];
rz(8.959788244774732) q[17];
sx q[17];
rz(5*pi/2) q[17];
sx q[17];
cx q[18],q[3];
rz(-2.212911375054119) q[3];
cx q[18],q[3];
rz(pi/2) q[18];
cx q[18],q[17];
x q[18];
cx q[13],q[18];
cx q[18],q[13];
rz(pi/4) q[13];
x q[18];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[3],q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/4) q[3];
cx q[3],q[15];
rz(-pi/4) q[15];
cx q[3],q[15];
rz(pi/4) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
sx q[15];
cx q[0],q[15];
x q[0];
x q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[1],q[15];
rz(-pi/4) q[15];
cx q[1],q[15];
rz(pi/4) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[19];
cx q[2],q[19];
cx q[19],q[2];
rz(5.066112751056649) q[19];
sx q[19];
rz(9.40600608899527) q[19];
sx q[19];
rz(14.076117054353755) q[19];
rz(pi) q[19];
x q[19];
rz(pi/2) q[19];
rz(-0.10613260551917336) q[19];
sx q[19];
rz(5.347672359681477) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(0.44248849424707387) q[2];
cx q[7],q[2];
rz(-0.44248849424707387) q[2];
cx q[7],q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/4) q[7];
cx q[7],q[16];
rz(-pi/4) q[16];
cx q[7],q[16];
rz(pi/4) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[1],q[16];
rz(1.4170056111947424) q[16];
cx q[1],q[16];
rz(0) q[1];
sx q[1];
rz(6.530467691982988) q[1];
sx q[1];
rz(3*pi) q[1];
rz(-pi/2) q[1];
rz(pi/4) q[16];
x q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(pi) q[20];
x q[20];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(-pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[21],q[8];
rz(0) q[21];
sx q[21];
rz(4.936736164894647) q[21];
sx q[21];
rz(3*pi) q[21];
rz(0) q[8];
sx q[8];
rz(1.3464491422849394) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[21],q[8];
rz(-pi/2) q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
rz(1.1254257221131065) q[21];
cx q[4],q[21];
rz(-1.9295722396295032) q[21];
sx q[21];
rz(0.9955776002705132) q[21];
sx q[21];
rz(3*pi) q[21];
cx q[4],q[21];
rz(0) q[21];
sx q[21];
rz(5.287607706909073) q[21];
sx q[21];
rz(10.228924478285776) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[4],q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/4) q[4];
cx q[4],q[2];
rz(-pi/4) q[2];
cx q[4],q[2];
rz(pi/4) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(3.62970558504943) q[4];
rz(-pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(-pi/2) q[8];
rz(4.031213307855425) q[8];
rz(-4.350141578962142) q[8];
sx q[8];
rz(4.50275274165343) q[8];
sx q[8];
rz(13.77491953973152) q[8];
rz(-pi/4) q[22];
cx q[20],q[22];
rz(0.8868097836456591) q[22];
cx q[20],q[22];
rz(pi) q[20];
x q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
cx q[20],q[5];
rz(1.135895984057076) q[22];
cx q[22],q[17];
rz(-1.135895984057076) q[17];
cx q[22],q[17];
rz(1.135895984057076) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[13],q[17];
rz(-pi/4) q[17];
cx q[13],q[17];
rz(5.4779973158776984) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/4) q[13];
rz(pi/4) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[17],q[15];
cx q[15],q[17];
cx q[17],q[15];
rz(-pi/2) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[17],q[4];
rz(3.8586008581754276) q[22];
sx q[22];
rz(3.3247667030512997) q[22];
rz(0.4510565528430896) q[22];
rz(-pi/2) q[22];
rz(3.3042060266352777) q[22];
rz(4.942508546679958) q[4];
cx q[17],q[4];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[17];
rz(2.660623043202181) q[17];
rz(1.3752889676256947) q[17];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[5],q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
cx q[18],q[20];
rz(pi/4) q[18];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
cx q[18],q[20];
rz(-pi/4) q[20];
cx q[18],q[20];
rz(2.8900126806653246) q[18];
rz(pi/4) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[20];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[24];
rz(pi/2) q[24];
sx q[24];
rz(pi/2) q[24];
rz(pi/4) q[24];
cx q[23],q[24];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
cx q[23],q[6];
rz(-pi/4) q[24];
rz(pi/2) q[24];
sx q[24];
rz(pi/2) q[24];
rz(-pi/2) q[24];
rz(pi/2) q[24];
rz(pi/2) q[24];
sx q[24];
rz(pi/2) q[24];
cx q[24],q[10];
cx q[10],q[24];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(-1.4169380593477052) q[24];
cx q[3],q[10];
cx q[10],q[3];
cx q[3],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[10],q[21];
rz(5.943996015006823) q[21];
cx q[10],q[21];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[10],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/4) q[10];
cx q[10],q[0];
rz(-pi/4) q[0];
cx q[10],q[0];
rz(pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(4.531146963045527) q[0];
id q[10];
rz(-pi/4) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(1.6504776507868955) q[21];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(1.8523634914109866) q[6];
cx q[23],q[6];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
cx q[14],q[23];
cx q[23],q[14];
rz(pi/4) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(0.7159231054204226) q[14];
cx q[11],q[14];
rz(-0.7159231054204226) q[14];
cx q[11],q[14];
rz(5.455974334991342) q[11];
rz(2.254777893051811) q[11];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/4) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[0],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(2.2898370027752035) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(-pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[23];
sx q[23];
rz(4.937749662656016) q[23];
sx q[23];
rz(5*pi/2) q[23];
cx q[21],q[23];
rz(-1.6504776507868955) q[23];
cx q[21],q[23];
rz(-pi/4) q[21];
rz(1.6504776507868955) q[23];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[7],q[21];
rz(4.184360661886022) q[21];
cx q[7],q[21];
rz(1.6296688023881374) q[7];
sx q[7];
rz(9.048056094929155) q[7];
sx q[7];
rz(14.351944968014092) q[7];
cx q[8],q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/4) q[8];
cx q[8],q[3];
rz(-pi/4) q[3];
cx q[8],q[3];
cx q[12],q[8];
rz(pi/4) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(0) q[3];
sx q[3];
rz(7.4898630954705645) q[3];
sx q[3];
rz(3*pi) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[16],q[3];
rz(-pi/4) q[3];
cx q[16],q[3];
cx q[16],q[18];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[18],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(1.1861094188018484) q[18];
sx q[18];
rz(7.113211738871461) q[18];
sx q[18];
rz(8.238668541967531) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
x q[18];
rz(pi/4) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(0.1927672108451162) q[8];
cx q[23],q[8];
rz(-0.1927672108451162) q[8];
cx q[23],q[8];
x q[23];
rz(4.937206104738477) q[23];
sx q[23];
rz(6.393842326225357) q[23];
sx q[23];
rz(13.1793864179971) q[23];
rz(-pi/2) q[23];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[9],q[24];
rz(-2.758367277373509) q[24];
sx q[24];
rz(2.881256929844584) q[24];
sx q[24];
rz(3*pi) q[24];
cx q[9],q[24];
rz(0) q[24];
sx q[24];
rz(3.401928377335002) q[24];
sx q[24];
rz(13.600083297490594) q[24];
cx q[24],q[6];
sx q[24];
cx q[2],q[24];
x q[2];
sx q[2];
cx q[20],q[2];
cx q[2],q[13];
rz(-pi/4) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(-pi/2) q[13];
x q[20];
rz(-0.5033956426596808) q[20];
rz(0.6235636467593321) q[24];
cx q[11],q[24];
rz(-2.254777893051811) q[24];
sx q[24];
rz(0.7485974848165085) q[24];
sx q[24];
rz(3*pi) q[24];
cx q[11],q[24];
rz(2.981932373571971) q[11];
cx q[11],q[20];
rz(-2.981932373571971) q[20];
sx q[20];
rz(0.572046760061149) q[20];
sx q[20];
rz(3*pi) q[20];
cx q[11],q[20];
rz(0) q[11];
sx q[11];
rz(7.513679537986498) q[11];
sx q[11];
rz(3*pi) q[11];
rz(pi/4) q[11];
rz(0) q[20];
sx q[20];
rz(5.711138547118438) q[20];
sx q[20];
rz(12.910105977001031) q[20];
cx q[2],q[20];
cx q[20],q[2];
cx q[2],q[20];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
cx q[20],q[1];
rz(2.4472780595338754) q[1];
cx q[20],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(0) q[24];
sx q[24];
rz(5.534587822363077) q[24];
sx q[24];
rz(11.055992207061859) q[24];
rz(-pi/2) q[24];
rz(pi/2) q[24];
sx q[24];
rz(pi/2) q[24];
cx q[24],q[3];
rz(4.0265491926081145) q[3];
cx q[24],q[3];
rz(pi/2) q[24];
sx q[24];
rz(pi/2) q[24];
rz(pi/2) q[24];
rz(0) q[24];
sx q[24];
rz(7.637810857888132) q[24];
sx q[24];
rz(3*pi) q[24];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
cx q[10],q[3];
cx q[3],q[10];
rz(-4.418724935349941) q[10];
sx q[10];
rz(7.291712699696148) q[10];
sx q[10];
rz(13.84350289611932) q[10];
rz(3.6347033321779616) q[10];
sx q[10];
rz(5.164714462550151) q[10];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[3],q[23];
rz(2.2810183785391014) q[23];
cx q[3],q[23];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
rz(pi/2) q[23];
rz(pi) q[23];
cx q[23],q[10];
rz(pi/2) q[10];
sx q[10];
rz(3.8009884528501123) q[10];
sx q[10];
rz(5*pi/2) q[10];
rz(0) q[23];
sx q[23];
rz(6.561291359521775) q[23];
sx q[23];
rz(3*pi) q[23];
cx q[23],q[10];
cx q[10],q[23];
cx q[23],q[10];
rz(pi/4) q[10];
rz(3.5550907142679993) q[10];
rz(1.4692295974307634) q[10];
rz(pi/4) q[23];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[5],q[6];
rz(pi/4) q[5];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[5],q[6];
rz(-pi/4) q[6];
cx q[5],q[6];
cx q[5],q[12];
cx q[12],q[5];
cx q[12],q[4];
rz(pi/4) q[12];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[12],q[4];
rz(-pi/4) q[4];
cx q[12],q[4];
rz(pi/4) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[12],q[4];
rz(1.1262300775008491) q[4];
cx q[12],q[4];
rz(pi) q[12];
x q[12];
cx q[12],q[20];
rz(0.438988254924928) q[20];
cx q[12],q[20];
rz(pi/2) q[20];
rz(-pi/2) q[20];
rz(0.4744114287165131) q[20];
rz(5.391793793573427) q[4];
sx q[4];
rz(6.264766837995925) q[4];
sx q[4];
rz(10.458024299326786) q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/4) q[4];
cx q[5],q[8];
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
cx q[15],q[6];
rz(pi/4) q[15];
cx q[15],q[19];
rz(-pi/4) q[19];
cx q[15],q[19];
rz(pi/4) q[15];
rz(pi/4) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/4) q[19];
cx q[19],q[16];
rz(-pi/4) q[16];
cx q[19],q[16];
rz(pi/4) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(1.7873717490244911) q[16];
rz(1.4204415514683828) q[16];
sx q[16];
rz(4.451431294105365) q[16];
rz(4.8472067699812955) q[16];
sx q[16];
rz(5.320428655051103) q[16];
sx q[16];
rz(14.688289026847087) q[16];
rz(pi/4) q[16];
rz(-pi/4) q[16];
sx q[16];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
x q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[13],q[6];
rz(6.2454616689499725) q[13];
sx q[13];
rz(6.892800063796504) q[13];
sx q[13];
rz(13.420178646827448) q[13];
rz(-4.242737211856472) q[13];
rz(pi/2) q[13];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[7],q[24];
rz(1.2777709603586542) q[8];
cx q[5],q[8];
rz(4.489051227912763) q[5];
sx q[5];
rz(7.586209569421797) q[5];
sx q[5];
rz(12.08685463794983) q[5];
cx q[5],q[6];
cx q[6],q[5];
cx q[5],q[6];
rz(4.26665500896178) q[5];
cx q[12],q[5];
cx q[5],q[12];
id q[12];
cx q[20],q[12];
rz(-0.4744114287165131) q[12];
cx q[20],q[12];
rz(0.4744114287165131) q[12];
rz(1.1512300327349099) q[20];
rz(2.234191848588497) q[5];
cx q[5],q[18];
rz(-2.234191848588497) q[18];
cx q[5],q[18];
rz(2.234191848588497) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/4) q[5];
cx q[8],q[21];
cx q[21],q[8];
rz(-pi/2) q[21];
cx q[19],q[21];
cx q[19],q[24];
rz(pi/2) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[24],q[19];
cx q[19],q[24];
rz(3.940544702907265) q[19];
sx q[19];
rz(4.478943671750601) q[19];
sx q[19];
rz(10.462675056105793) q[19];
rz(pi/2) q[19];
sx q[19];
rz(9.244813465523851) q[19];
sx q[19];
rz(5*pi/2) q[19];
rz(4.26952727200912) q[19];
sx q[19];
rz(1.912025597551363) q[19];
rz(pi/2) q[19];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[3],q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/4) q[3];
cx q[3],q[21];
rz(-pi/4) q[21];
cx q[3],q[21];
rz(pi/4) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(4.135601901108535) q[21];
sx q[21];
rz(7.042943175831009) q[21];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[8],q[17];
cx q[17],q[8];
cx q[8],q[17];
cx q[17],q[11];
cx q[11],q[17];
cx q[17],q[11];
cx q[8],q[4];
rz(-pi/4) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(-pi/2) q[4];
rz(0.3161965013997372) q[9];
sx q[9];
rz(4.145999444498851) q[9];
sx q[9];
rz(9.108581459369642) q[9];
rz(pi/2) q[9];
sx q[9];
rz(6.563923534480365) q[9];
sx q[9];
rz(5*pi/2) q[9];
rz(pi/4) q[9];
rz(-2.1587240874209694) q[9];
cx q[22],q[9];
rz(-3.3042060266352777) q[9];
sx q[9];
rz(1.8688673159607836) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[22],q[9];
rz(0.6834126971143031) q[22];
cx q[22],q[0];
rz(-0.6834126971143031) q[0];
cx q[22],q[0];
rz(0.6834126971143031) q[0];
rz(pi/4) q[0];
rz(-pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[1],q[0];
rz(1.8177442175456806) q[0];
cx q[1],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/4) q[1];
cx q[22],q[2];
rz(2.825514579678906) q[2];
cx q[22],q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/4) q[2];
cx q[2],q[11];
rz(-pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(-pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[2],q[14];
rz(3.2686096076548075) q[14];
cx q[2],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/4) q[14];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/4) q[2];
cx q[22],q[6];
rz(0) q[22];
sx q[22];
rz(4.031143973111709) q[22];
sx q[22];
rz(3*pi) q[22];
cx q[6],q[24];
rz(-3.344329381452119) q[6];
sx q[6];
rz(8.923384752759222) q[6];
sx q[6];
rz(12.7691073422215) q[6];
rz(5.241044042891025) q[6];
rz(2.8429129063235496) q[6];
cx q[8],q[1];
rz(-pi/4) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi/2) q[1];
x q[1];
rz(3.370181801202782) q[1];
rz(4.4987448918272035) q[1];
rz(pi) q[8];
rz(3.1049285645654914) q[8];
sx q[8];
rz(7.4478755905246485) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(0) q[9];
sx q[9];
rz(4.414317991218803) q[9];
sx q[9];
rz(14.887708074825627) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[15],q[9];
rz(5.6948819980568155) q[9];
cx q[15],q[9];
rz(-pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[15],q[13];
rz(0) q[13];
sx q[13];
rz(2.857532262337625) q[13];
sx q[13];
rz(3*pi) q[13];
rz(0) q[15];
sx q[15];
rz(3.4256530448419613) q[15];
sx q[15];
rz(3*pi) q[15];
cx q[15],q[13];
rz(-pi/2) q[13];
rz(4.242737211856472) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(2.6155170498612192) q[13];
rz(-pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
cx q[17],q[13];
rz(-2.6155170498612192) q[13];
cx q[17],q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(3.763946865503901) q[17];
rz(pi/2) q[17];
cx q[11],q[17];
rz(0) q[11];
sx q[11];
rz(0.10611967225780283) q[11];
sx q[11];
rz(3*pi) q[11];
rz(0) q[17];
sx q[17];
rz(0.10611967225780283) q[17];
sx q[17];
rz(3*pi) q[17];
cx q[11],q[17];
rz(-pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(4.469370088661812) q[11];
sx q[11];
rz(6.953287117170445) q[11];
rz(-pi/2) q[17];
rz(-3.763946865503901) q[17];
rz(pi/2) q[17];
cx q[18],q[17];
cx q[17],q[18];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[4],q[15];
cx q[15],q[4];
rz(pi/4) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[2],q[15];
rz(-pi/4) q[15];
cx q[2],q[15];
rz(pi/4) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(-pi/4) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[2],q[8];
cx q[21],q[4];
rz(2.5951051790988675) q[4];
cx q[21],q[4];
sx q[21];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[5],q[4];
rz(-pi/4) q[4];
cx q[5],q[4];
rz(pi/4) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(0.7965129031690965) q[4];
cx q[12],q[4];
rz(-0.7965129031690965) q[4];
cx q[12],q[4];
x q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(2.99500047646449) q[5];
rz(pi/2) q[5];
rz(6.249533742299071) q[8];
cx q[2],q[8];
cx q[2],q[12];
cx q[12],q[2];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(3.6626311949914596) q[12];
sx q[12];
rz(6.246065511364688) q[12];
rz(0.4333284009461297) q[12];
rz(4.088119556468942) q[12];
rz(-pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(-pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[7],q[9];
cx q[9],q[7];
cx q[0],q[9];
cx q[7],q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/4) q[7];
cx q[7],q[3];
rz(-pi/4) q[3];
cx q[7],q[3];
rz(pi/4) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[24],q[3];
rz(2.4952983599234098) q[3];
cx q[24],q[3];
rz(pi/2) q[24];
cx q[24],q[21];
rz(pi/2) q[21];
cx q[21],q[16];
rz(0.4066504253707488) q[16];
sx q[16];
rz(7.1046125654392895) q[16];
x q[21];
rz(0.3129482984730261) q[21];
rz(4.3445598766878994) q[21];
x q[24];
rz(pi/2) q[24];
sx q[24];
rz(pi/2) q[24];
cx q[23],q[24];
rz(-pi/4) q[24];
cx q[23],q[24];
rz(pi) q[23];
rz(pi/4) q[24];
rz(pi/2) q[24];
sx q[24];
rz(pi/2) q[24];
rz(pi/2) q[24];
rz(pi/2) q[24];
sx q[24];
rz(pi/2) q[24];
rz(pi/2) q[24];
rz(-1.2785951164022293) q[24];
sx q[24];
rz(6.274593990374617) q[24];
sx q[24];
rz(10.703373077171609) q[24];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
id q[3];
sx q[3];
rz(pi) q[7];
x q[7];
rz(1.8840239446521334) q[7];
cx q[6],q[7];
rz(-2.8429129063235496) q[7];
sx q[7];
rz(1.9862843714121599) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[6],q[7];
rz(1.9117633828449185) q[6];
cx q[6],q[18];
rz(-1.9117633828449185) q[18];
cx q[6],q[18];
rz(1.9117633828449185) q[18];
rz(pi) q[6];
x q[6];
rz(0) q[7];
sx q[7];
rz(4.296900935767426) q[7];
sx q[7];
rz(10.383666922440796) q[7];
rz(pi/2) q[7];
cx q[19],q[7];
cx q[7],q[19];
cx q[18],q[19];
rz(pi/4) q[18];
rz(-pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
cx q[19],q[23];
cx q[23],q[19];
cx q[19],q[23];
rz(4.329579450050858) q[19];
sx q[19];
rz(5.414913152311026) q[19];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[9],q[0];
cx q[0],q[9];
rz(2.006783942706015) q[0];
cx q[0],q[22];
rz(-2.006783942706015) q[22];
cx q[0],q[22];
rz(-0.7941213659695396) q[0];
cx q[1],q[0];
rz(-4.4987448918272035) q[0];
sx q[0];
rz(0.3597542781044418) q[0];
sx q[0];
rz(3*pi) q[0];
cx q[1],q[0];
rz(0) q[0];
sx q[0];
rz(5.923431029075145) q[0];
sx q[0];
rz(14.717644218566122) q[0];
rz(0) q[0];
sx q[0];
rz(6.934992182282685) q[0];
sx q[0];
rz(3*pi) q[0];
rz(0) q[0];
sx q[0];
rz(7.429214891727006) q[0];
sx q[0];
rz(3*pi) q[0];
rz(6.201207309232451) q[0];
rz(4.732760184797389) q[0];
cx q[0],q[10];
cx q[1],q[11];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-4.732760184797389) q[10];
sx q[10];
rz(3.123029406966329) q[10];
sx q[10];
rz(3*pi) q[10];
cx q[0],q[10];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(0) q[10];
sx q[10];
rz(3.1601559002132573) q[10];
sx q[10];
rz(12.688308548136005) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(1.3350944640138784) q[10];
rz(2.006783942706015) q[22];
rz(pi/4) q[22];
cx q[22],q[20];
rz(-1.1512300327349099) q[20];
cx q[22],q[20];
rz(1.6010126765428436) q[20];
cx q[22],q[20];
rz(-1.6010126765428436) q[20];
cx q[22],q[20];
rz(-1.3578576476826152) q[20];
sx q[20];
rz(5.509918605797981) q[20];
rz(pi/4) q[20];
rz(pi/2) q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(pi/4) q[22];
cx q[6],q[22];
rz(-pi/4) q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(-pi/2) q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(0) q[6];
sx q[6];
rz(3.3688983173462477) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[24],q[6];
rz(0) q[6];
sx q[6];
rz(2.9142869898333386) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[24],q[6];
cx q[19],q[6];
x q[24];
rz(pi/2) q[24];
rz(pi/2) q[24];
sx q[24];
rz(pi/2) q[24];
rz(pi/4) q[24];
cx q[6],q[19];
rz(pi/2) q[19];
rz(pi/4) q[6];
cx q[7],q[1];
rz(2.1381047079644517) q[1];
cx q[7],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[1],q[2];
rz(2.9897419142819968) q[2];
cx q[1],q[2];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/4) q[1];
rz(-0.9674331691273945) q[1];
cx q[12],q[1];
rz(-4.088119556468942) q[1];
sx q[1];
rz(2.94083956296049) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[12],q[1];
rz(0) q[1];
sx q[1];
rz(3.342345744219096) q[1];
sx q[1];
rz(14.480330686365715) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi/4) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(1.841196358804111) q[7];
sx q[7];
rz(9.25142263164426) q[7];
sx q[7];
rz(15.287595156582237) q[7];
sx q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[13],q[9];
rz(2.1796532321929254) q[13];
cx q[17],q[13];
cx q[13],q[17];
rz(pi/4) q[13];
cx q[13],q[15];
rz(-pi/4) q[15];
cx q[13],q[15];
rz(1.2404329829621599) q[13];
rz(pi/4) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(0.183481955856867) q[17];
cx q[21],q[17];
rz(-4.3445598766878994) q[17];
sx q[17];
rz(0.37904827267731056) q[17];
sx q[17];
rz(3*pi) q[17];
cx q[21],q[17];
rz(0) q[17];
sx q[17];
rz(5.904137034502275) q[17];
sx q[17];
rz(13.585855881600413) q[17];
cx q[17],q[13];
rz(-1.2404329829621599) q[13];
cx q[17],q[13];
rz(-3.7185387473487745) q[17];
rz(pi/2) q[17];
cx q[18],q[17];
rz(0) q[17];
sx q[17];
rz(0.6874289926146901) q[17];
sx q[17];
rz(3*pi) q[17];
rz(0) q[18];
sx q[18];
rz(5.5957563145648965) q[18];
sx q[18];
rz(3*pi) q[18];
cx q[18],q[17];
rz(-pi/2) q[17];
rz(3.7185387473487745) q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(-pi/2) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
sx q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
rz(pi/2) q[18];
rz(-0.16151505227956087) q[18];
rz(pi) q[21];
rz(pi/2) q[21];
cx q[0],q[21];
cx q[21],q[0];
rz(1.7833226209229753) q[0];
cx q[0],q[18];
rz(-1.7833226209229753) q[18];
sx q[18];
rz(0.5396961236555908) q[18];
sx q[18];
rz(3*pi) q[18];
cx q[0],q[18];
rz(0.9771722341408615) q[0];
rz(0) q[18];
sx q[18];
rz(5.743489183523995) q[18];
sx q[18];
rz(11.369615633971915) q[18];
cx q[18],q[24];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(0) q[21];
sx q[21];
rz(4.857659658837211) q[21];
sx q[21];
rz(3*pi) q[21];
rz(-pi/4) q[24];
rz(pi/2) q[24];
sx q[24];
rz(pi/2) q[24];
rz(-pi/2) q[24];
rz(5.501955306028123) q[24];
rz(pi/2) q[24];
cx q[4],q[15];
cx q[15],q[4];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[20],q[15];
rz(-pi/4) q[15];
cx q[20],q[15];
rz(pi/4) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(1.9661167824045498) q[20];
rz(pi) q[4];
x q[4];
rz(4.79149025082032) q[4];
cx q[4],q[0];
rz(-4.79149025082032) q[0];
sx q[0];
rz(2.6478653319365306) q[0];
sx q[0];
rz(3*pi) q[0];
cx q[4],q[0];
rz(0) q[0];
sx q[0];
rz(3.6353199752430556) q[0];
sx q[0];
rz(13.239095977448837) q[0];
rz(4.914460105732862) q[0];
cx q[6],q[15];
rz(-pi/4) q[15];
cx q[6],q[15];
rz(pi/4) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[14];
rz(-pi/4) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-pi/2) q[14];
rz(3.741196321636453) q[14];
rz(1.4545212396044551) q[14];
cx q[14],q[11];
rz(-1.4545212396044551) q[11];
cx q[14],q[11];
rz(1.4545212396044551) q[11];
rz(pi) q[11];
cx q[11],q[13];
cx q[13],q[11];
rz(2.1405550340094583) q[13];
cx q[11],q[13];
rz(-2.1405550340094583) q[13];
cx q[11],q[13];
rz(pi/2) q[11];
rz(pi/2) q[13];
sx q[13];
rz(5.578144930051519) q[13];
sx q[13];
rz(5*pi/2) q[13];
rz(pi) q[13];
x q[13];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
cx q[2],q[14];
cx q[14],q[2];
cx q[2],q[14];
cx q[20],q[14];
rz(-1.9661167824045498) q[14];
cx q[20],q[14];
rz(1.9661167824045498) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(-pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[5];
rz(0) q[5];
sx q[5];
rz(0.04975702578708052) q[5];
sx q[5];
rz(3*pi) q[5];
rz(0) q[9];
sx q[9];
rz(0.04975702578708052) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[9],q[5];
rz(-pi/2) q[5];
rz(-2.99500047646449) q[5];
rz(-3.0171824788075825) q[5];
rz(pi/2) q[5];
cx q[8],q[5];
rz(0) q[5];
sx q[5];
rz(2.698311181235974) q[5];
sx q[5];
rz(3*pi) q[5];
rz(0) q[8];
sx q[8];
rz(3.584874125943612) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[8],q[5];
rz(-pi/2) q[5];
rz(3.0171824788075825) q[5];
rz(-3.0859536664238467) q[5];
sx q[5];
rz(5.686924391910815) q[5];
sx q[5];
rz(12.510731627193227) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[5],q[23];
rz(0.03596193959546988) q[23];
cx q[5],q[23];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/4) q[5];
cx q[2],q[5];
rz(0) q[2];
sx q[2];
rz(4.936577988390424) q[2];
sx q[2];
rz(3*pi) q[2];
rz(-pi/4) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-pi/2) q[5];
rz(-pi/2) q[5];
rz(pi/4) q[5];
cx q[5],q[1];
rz(-pi/4) q[1];
cx q[5],q[1];
rz(pi/4) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(2.4515473254578786) q[8];
cx q[8],q[16];
rz(-2.4515473254578786) q[16];
cx q[8],q[16];
rz(2.4515473254578786) q[16];
cx q[16],q[10];
rz(-1.3350944640138784) q[10];
cx q[16],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[10],q[21];
rz(1.6358044577926505) q[16];
sx q[16];
rz(7.680098175426564) q[16];
sx q[16];
rz(14.180038878910771) q[16];
rz(-pi/2) q[16];
cx q[17],q[16];
rz(pi/2) q[16];
rz(-pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[16],q[24];
rz(0) q[16];
sx q[16];
rz(0.06519077536797813) q[16];
sx q[16];
rz(3*pi) q[16];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(1.841990185828582) q[17];
cx q[11],q[17];
rz(-1.841990185828582) q[17];
cx q[11],q[17];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(0) q[21];
sx q[21];
rz(1.4255256483423746) q[21];
sx q[21];
rz(3*pi) q[21];
cx q[10],q[21];
rz(-4.9295395198044165) q[10];
rz(pi/2) q[10];
rz(5.786745450377697) q[21];
rz(3.7645485794266147) q[21];
rz(0) q[24];
sx q[24];
rz(0.06519077536797813) q[24];
sx q[24];
rz(3*pi) q[24];
cx q[16],q[24];
rz(-pi/2) q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(pi/2) q[16];
rz(-pi/2) q[24];
rz(-5.501955306028123) q[24];
rz(pi/4) q[8];
cx q[8],q[22];
rz(-pi/4) q[22];
cx q[8],q[22];
rz(pi/4) q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(0.9061503778697806) q[22];
cx q[22],q[23];
rz(-0.9061503778697806) q[23];
cx q[22],q[23];
rz(1.8794259290006003) q[22];
cx q[21],q[22];
rz(-3.7645485794266147) q[22];
sx q[22];
rz(1.4071251244233918) q[22];
sx q[22];
rz(3*pi) q[22];
cx q[21],q[22];
rz(2.6023216541534535) q[21];
rz(0) q[22];
sx q[22];
rz(4.876060182756195) q[22];
sx q[22];
rz(11.309900611195394) q[22];
cx q[18],q[22];
cx q[22],q[18];
rz(0.9061503778697806) q[23];
cx q[23],q[2];
rz(0) q[2];
sx q[2];
rz(1.3466073187891618) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[23],q[2];
cx q[2],q[14];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(0.8065241168833558) q[23];
cx q[4],q[23];
rz(-0.8065241168833558) q[23];
cx q[4],q[23];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[19];
cx q[19],q[8];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(-4.575845131074142) q[19];
sx q[19];
rz(4.482372257381966) q[19];
sx q[19];
rz(14.000623091843522) q[19];
cx q[8],q[21];
rz(-2.6023216541534535) q[21];
cx q[8],q[21];
rz(-pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
cx q[9],q[3];
rz(2.2136523592712165) q[3];
x q[9];
cx q[3],q[9];
rz(-2.2136523592712165) q[9];
cx q[3],q[9];
x q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[7],q[3];
rz(0.4567838408817965) q[3];
cx q[7],q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/4) q[3];
cx q[20],q[3];
x q[20];
rz(-pi/4) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[6],q[3];
rz(5.11203613673437) q[3];
cx q[6],q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(2.5588959314342357) q[7];
sx q[7];
rz(5.730780408215964) q[7];
sx q[7];
rz(9.831943743195897) q[7];
rz(pi/4) q[7];
cx q[7],q[15];
rz(-pi/4) q[15];
cx q[7],q[15];
rz(pi/4) q[15];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(2.2136523592712165) q[9];
rz(-pi/4) q[9];
x q[9];
rz(-pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[10];
rz(0) q[10];
sx q[10];
rz(2.132739336215285) q[10];
sx q[10];
rz(3*pi) q[10];
rz(0) q[9];
sx q[9];
rz(4.150445970964301) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[9],q[10];
rz(-pi/2) q[10];
rz(4.9295395198044165) q[10];
rz(pi/4) q[10];
rz(-pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[10],q[9];
rz(-pi/4) q[9];
cx q[10],q[9];
rz(pi/4) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
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
measure q[22] -> c[22];
measure q[23] -> c[23];
measure q[24] -> c[24];
