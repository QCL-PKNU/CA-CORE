OPENQASM 2.0;
include "qelib1.inc";
qreg q[13];
creg c[13];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[1];
sx q[1];
rz(8.66342942784895) q[1];
sx q[1];
rz(5*pi/2) q[1];
x q[1];
rz(2.333869885848389) q[2];
rz(2.8422553249836118) q[3];
rz(pi/2) q[3];
cx q[0],q[3];
rz(0) q[0];
sx q[0];
rz(2.4309519566127653) q[0];
sx q[0];
rz(3*pi) q[0];
rz(0) q[3];
sx q[3];
rz(2.4309519566127653) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[0],q[3];
rz(-pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(0) q[0];
sx q[0];
rz(8.467816925708318) q[0];
sx q[0];
rz(3*pi) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi/2) q[3];
rz(-2.8422553249836118) q[3];
rz(0.8262169512831711) q[3];
rz(-pi/2) q[3];
cx q[4],q[5];
rz(-2.9979222610794984) q[4];
sx q[4];
rz(7.481950336574907) q[4];
sx q[4];
rz(12.422700221848878) q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(-pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-pi/4) q[6];
rz(0) q[6];
sx q[6];
rz(4.1368226265557455) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[2],q[7];
rz(-2.333869885848389) q[7];
cx q[2],q[7];
rz(pi/2) q[2];
cx q[2],q[3];
rz(1.9977006112899311) q[2];
rz(pi/2) q[3];
rz(-pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(2.333869885848389) q[7];
rz(-0.31372241181403515) q[7];
rz(pi/2) q[7];
cx q[5],q[7];
rz(0) q[5];
sx q[5];
rz(5.100650765982423) q[5];
sx q[5];
rz(3*pi) q[5];
rz(0) q[7];
sx q[7];
rz(1.1825345411971633) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[5],q[7];
rz(-pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(4.272010981963554) q[5];
rz(1.2993020833844555) q[5];
rz(1.9789124833675427) q[5];
rz(-pi/2) q[7];
rz(0.31372241181403515) q[7];
rz(5.814917727225776) q[7];
sx q[7];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/4) q[8];
cx q[9],q[8];
rz(-pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(-pi/2) q[8];
cx q[8],q[6];
rz(0) q[6];
sx q[6];
rz(2.1463626806238407) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[8],q[6];
rz(pi/4) q[6];
cx q[6],q[0];
rz(-pi/4) q[0];
cx q[6],q[0];
rz(pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(4.164265909184048) q[0];
rz(pi/2) q[0];
cx q[3],q[0];
rz(0) q[0];
sx q[0];
rz(2.0803760484770684) q[0];
sx q[0];
rz(3*pi) q[0];
rz(0) q[3];
sx q[3];
rz(2.0803760484770684) q[3];
sx q[3];
rz(3*pi) q[3];
cx q[3],q[0];
rz(-pi/2) q[0];
rz(-4.164265909184048) q[0];
rz(3.2282042805748743) q[0];
x q[0];
rz(1.0235432251150727) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(1.4274806266917346) q[6];
rz(2.0042300424397377) q[6];
rz(2.6790396884134826) q[6];
cx q[8],q[1];
rz(5.513135441883934) q[1];
cx q[8],q[1];
rz(5.815332865394353) q[1];
sx q[1];
rz(7.4224314016456905) q[1];
sx q[1];
rz(12.557906143031095) q[1];
rz(pi) q[1];
x q[1];
rz(1.6501157885659576) q[8];
sx q[8];
rz(5.6959579747300735) q[8];
sx q[8];
rz(14.633647825850225) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(2.4180875402913466) q[9];
rz(pi/4) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(-1.252827148261268) q[11];
sx q[11];
rz(2.244024442906267) q[11];
cx q[11],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/4) q[11];
cx q[11],q[10];
rz(-pi/4) q[10];
cx q[11],q[10];
rz(pi/4) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi) q[10];
rz(pi/2) q[10];
cx q[10],q[7];
x q[10];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/4) q[7];
cx q[10],q[7];
rz(pi/2) q[10];
sx q[10];
rz(6.131945730524239) q[10];
sx q[10];
rz(5*pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(-pi/4) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(-pi/2) q[7];
rz(pi) q[12];
x q[12];
cx q[9],q[12];
rz(-2.4180875402913466) q[12];
cx q[9],q[12];
rz(2.4180875402913466) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[11],q[12];
rz(0.15984014850934558) q[12];
cx q[11],q[12];
cx q[11],q[2];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-1.9977006112899311) q[2];
cx q[11],q[2];
rz(2.8770241917649013) q[11];
sx q[11];
rz(4.54712187184047) q[11];
sx q[11];
rz(15.328787876732278) q[11];
rz(pi/2) q[11];
rz(-pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/4) q[11];
cx q[2],q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[9];
cx q[4],q[9];
cx q[9],q[4];
rz(5.751897577419729) q[4];
sx q[4];
rz(5.624709951337644) q[4];
sx q[4];
rz(14.098461705718606) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[3],q[4];
rz(-pi/2) q[3];
cx q[1],q[3];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/4) q[1];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
sx q[4];
rz(0.6005825966580488) q[4];
sx q[4];
rz(3.912719418840332) q[4];
sx q[4];
rz(15.266024531402511) q[4];
rz(pi/4) q[4];
cx q[4],q[0];
rz(-pi/4) q[0];
cx q[4],q[0];
rz(pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[12];
rz(-0.7236170863376876) q[12];
cx q[6],q[12];
rz(-2.6790396884134826) q[12];
sx q[12];
rz(0.560481533875143) q[12];
sx q[12];
rz(3*pi) q[12];
cx q[6],q[12];
rz(0) q[12];
sx q[12];
rz(5.722703773304444) q[12];
sx q[12];
rz(12.82743473552055) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/4) q[12];
cx q[2],q[12];
rz(-pi/4) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-pi/2) q[12];
rz(-pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[2];
cx q[3],q[2];
cx q[2],q[3];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(4.443977462060887) q[2];
rz(pi/2) q[2];
cx q[7],q[6];
cx q[6],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(2.9984174548057334) q[10];
rz(0.7867988059108697) q[6];
rz(pi/2) q[6];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(-pi/2) q[7];
cx q[3],q[7];
rz(-pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[3],q[0];
rz(3.6871443354730276) q[0];
cx q[3],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(4.135809961903652) q[0];
rz(6.129777619436899) q[0];
rz(pi/2) q[0];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
cx q[3],q[11];
rz(0.37397346087419037) q[11];
cx q[3],q[11];
rz(2.465062140276551) q[11];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(2.654370357088225) q[3];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[1],q[7];
rz(-pi/4) q[7];
cx q[1],q[7];
rz(-pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/4) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(0.5433363858536849) q[7];
sx q[7];
rz(5.225972114496155) q[7];
sx q[7];
rz(11.267895860965492) q[7];
rz(-pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[0];
rz(0) q[0];
sx q[0];
rz(1.6170423671384486) q[0];
sx q[0];
rz(3*pi) q[0];
rz(0) q[7];
sx q[7];
rz(1.6170423671384486) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[7],q[0];
rz(-pi/2) q[0];
rz(-6.129777619436899) q[0];
rz(0) q[0];
sx q[0];
rz(7.155841551080486) q[0];
sx q[0];
rz(3*pi) q[0];
rz(pi/2) q[0];
rz(-pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
id q[7];
rz(2.859747589787839) q[9];
rz(2.2714883148724088) q[9];
cx q[9],q[5];
rz(-2.2714883148724088) q[5];
sx q[5];
rz(2.111987481811095) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[9],q[5];
rz(0) q[5];
sx q[5];
rz(4.171197825368491) q[5];
sx q[5];
rz(9.717353792274245) q[5];
x q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(2.184137421304833) q[5];
cx q[9],q[8];
cx q[8],q[9];
cx q[9],q[8];
cx q[8],q[5];
rz(-2.184137421304833) q[5];
cx q[8],q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[5],q[2];
rz(0) q[2];
sx q[2];
rz(1.3450701835105052) q[2];
sx q[2];
rz(3*pi) q[2];
rz(0) q[5];
sx q[5];
rz(1.3450701835105052) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[5],q[2];
rz(-pi/2) q[2];
rz(-4.443977462060887) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(-pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(-pi/2) q[5];
rz(-2.21470367991909) q[8];
cx q[10],q[8];
rz(-2.9984174548057334) q[8];
sx q[8];
rz(0.8314240618855169) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[10],q[8];
cx q[10],q[2];
rz(2.9127979644154607) q[2];
cx q[10],q[2];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(5.444710448012426) q[10];
sx q[10];
cx q[0],q[10];
x q[0];
rz(1.770220433917035) q[0];
sx q[0];
rz(7.960630961240605) q[0];
sx q[0];
rz(13.758344854267186) q[0];
rz(-pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
sx q[10];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(6.197617214502207) q[2];
rz(pi/2) q[2];
cx q[1],q[2];
rz(0) q[1];
sx q[1];
rz(0.08700695953484194) q[1];
sx q[1];
rz(3*pi) q[1];
rz(0) q[2];
sx q[2];
rz(0.08700695953484194) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[1],q[2];
rz(-pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(0.6674417077114959) q[1];
rz(-pi/2) q[2];
rz(-6.197617214502207) q[2];
rz(-pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(0) q[8];
sx q[8];
rz(5.451761245294069) q[8];
sx q[8];
rz(14.637899095494202) q[8];
rz(pi/2) q[8];
cx q[4],q[8];
cx q[8],q[4];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(-pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(-pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[12];
rz(5.873124515035231) q[12];
cx q[9],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(5.212718093094014) q[12];
sx q[12];
rz(6.91594953254444) q[12];
sx q[12];
rz(13.782879172997317) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(1.6898489952218394) q[12];
rz(-pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[12],q[2];
rz(3.5485484151094533) q[2];
cx q[12],q[2];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(2.9450337671794697) q[12];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(5.882830896609739) q[2];
id q[2];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(-pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[6];
rz(0) q[6];
sx q[6];
rz(2.0425737741210757) q[6];
sx q[6];
rz(3*pi) q[6];
rz(0) q[9];
sx q[9];
rz(2.0425737741210757) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[9],q[6];
rz(-pi/2) q[6];
rz(-0.7867988059108697) q[6];
id q[6];
rz(pi/2) q[6];
cx q[6],q[11];
rz(-2.465062140276551) q[11];
cx q[6],q[11];
cx q[1],q[6];
rz(pi) q[11];
x q[11];
rz(3.504660557341532) q[11];
sx q[11];
rz(5.746158335882303) q[11];
sx q[11];
rz(10.699192876797431) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(5.528833600413067) q[11];
rz(-pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(1.5477749710791728) q[6];
cx q[1],q[6];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(0.15775868923436054) q[6];
cx q[7],q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/4) q[7];
cx q[7],q[1];
rz(-pi/4) q[1];
cx q[7],q[1];
rz(pi/4) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(0) q[1];
sx q[1];
rz(7.999427595890315) q[1];
sx q[1];
rz(3*pi) q[1];
rz(pi/4) q[1];
rz(pi/2) q[7];
rz(-pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
cx q[9],q[5];
rz(pi/2) q[5];
rz(0.5848789409669859) q[5];
cx q[5],q[4];
rz(-0.5848789409669859) q[4];
cx q[5],q[4];
rz(0.5848789409669859) q[4];
rz(-pi/2) q[4];
cx q[4],q[12];
rz(-2.9450337671794697) q[12];
cx q[4],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi) q[12];
rz(pi/2) q[12];
cx q[12],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(-pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[11],q[10];
rz(4.89335216332494) q[10];
cx q[11],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(2.522688603207621) q[10];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(0) q[11];
sx q[11];
rz(5.461304792628862) q[11];
sx q[11];
rz(3*pi) q[11];
x q[12];
rz(0) q[4];
sx q[4];
rz(5.784646031979248) q[4];
sx q[4];
rz(3*pi) q[4];
sx q[4];
rz(pi) q[5];
x q[5];
rz(4.519446233375013) q[5];
sx q[5];
rz(5.796674658989856) q[5];
sx q[5];
rz(9.689518236668848) q[5];
rz(4.896067870133048) q[5];
cx q[5],q[6];
rz(-4.896067870133048) q[6];
sx q[6];
rz(2.52250730974744) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[5],q[6];
rz(pi/2) q[5];
rz(0) q[6];
sx q[6];
rz(3.760677997432146) q[6];
sx q[6];
rz(14.163087141668067) q[6];
cx q[7],q[4];
rz(pi/2) q[4];
sx q[4];
rz(6.580866462340626) q[4];
sx q[4];
rz(5*pi/2) q[4];
rz(0.20879569154375277) q[4];
sx q[4];
rz(3.9462936923971923) q[4];
cx q[4],q[10];
rz(-2.522688603207621) q[10];
cx q[4],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/4) q[4];
x q[7];
rz(-pi/2) q[7];
rz(2.3772906401279714) q[9];
cx q[9],q[3];
rz(-2.654370357088225) q[3];
cx q[9],q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(8.34541779136428) q[3];
sx q[3];
rz(5*pi/2) q[3];
rz(1.3230085941663265) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[3],q[5];
cx q[5],q[3];
cx q[3],q[12];
rz(5.680918153214258) q[12];
cx q[3],q[12];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[8];
cx q[8],q[9];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(4.0212101808168015) q[8];
sx q[8];
rz(4.541747872858277) q[8];
cx q[8],q[2];
rz(3.8115027198147393) q[2];
cx q[8],q[2];
cx q[5],q[2];
cx q[2],q[5];
cx q[5],q[2];
cx q[2],q[3];
rz(0.22229646043568357) q[3];
cx q[2],q[3];
rz(3.046986225211617) q[3];
sx q[3];
rz(3.2215018544377094) q[3];
sx q[3];
rz(0) q[5];
sx q[5];
rz(5.817098664572864) q[5];
sx q[5];
rz(3*pi) q[5];
x q[8];
rz(1.585044101583853) q[8];
cx q[12],q[8];
rz(-1.585044101583853) q[8];
cx q[12],q[8];
cx q[2],q[12];
rz(0) q[12];
sx q[12];
rz(5.099078096350301) q[12];
sx q[12];
rz(3*pi) q[12];
rz(pi/4) q[2];
cx q[2],q[10];
rz(-pi/4) q[10];
cx q[2],q[10];
rz(pi/4) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(0.7580993088049426) q[10];
rz(-pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(1.9181855002432577) q[8];
rz(6.0758919137140435) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[6],q[9];
rz(2.3915733237389216) q[9];
cx q[6],q[9];
cx q[6],q[7];
rz(0) q[6];
sx q[6];
rz(6.915016402722592) q[6];
sx q[6];
rz(3*pi) q[6];
sx q[6];
rz(pi/2) q[7];
rz(0.5755787960261967) q[7];
cx q[7],q[1];
rz(-0.5755787960261967) q[1];
cx q[7],q[1];
rz(0.5755787960261967) q[1];
rz(pi/2) q[1];
cx q[1],q[6];
x q[1];
rz(pi/4) q[1];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(7.951041480487893) q[7];
sx q[7];
rz(5*pi/2) q[7];
rz(pi/4) q[7];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(-pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[0],q[9];
rz(3.8899312586786055) q[9];
cx q[0],q[9];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(1.0604246326496551) q[0];
sx q[0];
rz(8.085093406502901) q[0];
sx q[0];
rz(8.364353328119725) q[0];
rz(2.5245324816517103) q[0];
cx q[0],q[8];
rz(-2.5245324816517103) q[8];
sx q[8];
rz(1.9803712798253534) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[0],q[8];
cx q[0],q[6];
rz(3.8144052393312675) q[6];
cx q[0],q[6];
rz(pi/4) q[0];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
x q[6];
rz(0) q[8];
sx q[8];
rz(4.302814027354232) q[8];
sx q[8];
rz(10.031124942177831) q[8];
cx q[8],q[12];
rz(0) q[12];
sx q[12];
rz(1.1841072108292854) q[12];
sx q[12];
rz(3*pi) q[12];
cx q[8],q[12];
cx q[8],q[12];
rz(2.920199629018233) q[12];
cx q[8],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[6],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/4) q[6];
cx q[6],q[12];
rz(-pi/4) q[12];
cx q[6],q[12];
rz(pi/4) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
cx q[9],q[5];
rz(0) q[5];
sx q[5];
rz(0.46608664260672183) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[9],q[5];
x q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[1],q[5];
rz(-pi/4) q[5];
cx q[1],q[5];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[0],q[1];
rz(-pi/4) q[1];
cx q[0],q[1];
rz(pi/4) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/4) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-0.426639155345818) q[5];
sx q[5];
rz(2.5786710751926396) q[5];
x q[5];
cx q[9],q[11];
rz(0) q[11];
sx q[11];
rz(0.8218805145507244) q[11];
sx q[11];
rz(3*pi) q[11];
cx q[9],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[4],q[11];
rz(-pi/4) q[11];
cx q[4],q[11];
rz(pi/4) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(2.621886092779461) q[11];
cx q[11],q[8];
rz(-pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[2],q[4];
rz(0.7650575541481122) q[4];
cx q[2],q[4];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
cx q[4],q[1];
rz(5.284114035435483) q[1];
cx q[4],q[1];
cx q[7],q[2];
rz(-pi/4) q[2];
cx q[7],q[2];
rz(pi/4) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(3.917778275124599) q[9];
cx q[9],q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(2.165153615316254) q[3];
cx q[0],q[3];
rz(-2.165153615316254) q[3];
cx q[0],q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(1.0476928078215528) q[9];
cx q[10],q[9];
rz(-1.0476928078215528) q[9];
cx q[10],q[9];
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