OPENQASM 2.0;
include "qelib1.inc";
qreg q[14];
creg c[14];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(1.799414689536597) q[2];
rz(-2.156292300158694) q[3];
rz(pi/2) q[3];
cx q[1],q[3];
rz(0) q[1];
sx q[1];
rz(4.336362782499238) q[1];
sx q[1];
rz(3*pi) q[1];
rz(0) q[3];
sx q[3];
rz(1.946822524680348) q[3];
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
rz(pi/2) q[1];
sx q[1];
rz(3.8286254139802365) q[1];
sx q[1];
rz(5*pi/2) q[1];
rz(0) q[1];
sx q[1];
rz(3.7762456527537807) q[1];
sx q[1];
rz(3*pi) q[1];
rz(-pi/2) q[3];
rz(2.156292300158694) q[3];
rz(pi) q[3];
x q[3];
rz(2.2922639413446544) q[3];
cx q[4],q[0];
rz(3.6068158429029853) q[0];
cx q[4],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi/2) q[0];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[4],q[5];
rz(pi/4) q[4];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[4],q[5];
rz(-pi/4) q[5];
cx q[4],q[5];
rz(pi/4) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(0) q[6];
sx q[6];
rz(3.1690657581930917) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[7],q[6];
rz(0) q[6];
sx q[6];
rz(3.1141195489864946) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[7],q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/4) q[7];
rz(pi/2) q[8];
rz(0.1095944783286833) q[9];
cx q[2],q[9];
rz(-1.799414689536597) q[9];
sx q[9];
rz(0.7703447497461364) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[2],q[9];
rz(0.4376751243991031) q[2];
rz(0) q[9];
sx q[9];
rz(5.512840557433449) q[9];
sx q[9];
rz(11.114598171977294) q[9];
rz(-pi/4) q[9];
cx q[3],q[9];
rz(-2.2922639413446544) q[9];
cx q[3],q[9];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(2.2922639413446544) q[9];
rz(pi/4) q[9];
rz(-0.5933640712419662) q[10];
sx q[10];
rz(2.644179225162213) q[10];
rz(-2.8554714739766176) q[10];
sx q[10];
rz(5.288281633575636) q[10];
sx q[10];
rz(12.280249434745997) q[10];
rz(5.994259441832025) q[10];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
cx q[11],q[0];
rz(pi/2) q[0];
cx q[0],q[1];
rz(0) q[1];
sx q[1];
rz(2.5069396544258056) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[0],q[1];
rz(pi/2) q[0];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(1.4186130271482902) q[11];
cx q[3],q[0];
cx q[0],q[3];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(2.4225839624230527) q[3];
rz(3.896034652853113) q[3];
cx q[5],q[11];
rz(-1.4186130271482902) q[11];
cx q[5],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
id q[11];
rz(0.3833381552868178) q[11];
sx q[11];
rz(5.097954057977788) q[11];
sx q[11];
rz(9.041439805482561) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(0.3119468955173892) q[5];
rz(pi/2) q[5];
rz(pi/4) q[12];
cx q[2],q[12];
rz(-0.4376751243991031) q[12];
cx q[2],q[12];
rz(0.4376751243991031) q[12];
rz(-0.2092197356399521) q[12];
cx q[10],q[12];
rz(-5.994259441832025) q[12];
sx q[12];
rz(2.71124433643202) q[12];
sx q[12];
rz(3*pi) q[12];
cx q[10],q[12];
rz(pi) q[10];
rz(0) q[12];
sx q[12];
rz(3.5719409707475664) q[12];
sx q[12];
rz(15.628257138241356) q[12];
rz(0.25630954042128223) q[12];
sx q[12];
rz(5.421215557660661) q[12];
sx q[12];
rz(14.642189913647844) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[4],q[2];
rz(0.5377395257576377) q[2];
cx q[4],q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(5.484025033838212) q[2];
sx q[2];
rz(8.13380864285655) q[2];
sx q[2];
rz(14.970480809996603) q[2];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[9],q[4];
rz(-pi/4) q[4];
cx q[9],q[4];
rz(pi/4) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(0) q[4];
sx q[4];
rz(6.409363280055748) q[4];
sx q[4];
rz(3*pi) q[4];
rz(0.2672854106251273) q[4];
rz(1.8491656776927452) q[4];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(-pi/2) q[9];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[13],q[8];
cx q[8],q[13];
rz(pi/2) q[13];
cx q[6],q[13];
cx q[13],q[6];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(2.4750519212854956) q[13];
rz(2.0710373688121315) q[6];
cx q[6],q[7];
rz(-2.0710373688121315) q[7];
cx q[6],q[7];
rz(-pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[6],q[5];
rz(0) q[5];
sx q[5];
rz(1.4189608552841615) q[5];
sx q[5];
rz(3*pi) q[5];
rz(0) q[6];
sx q[6];
rz(1.4189608552841615) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[6],q[5];
rz(-pi/2) q[5];
rz(-0.3119468955173892) q[5];
rz(2.146193620051242) q[5];
rz(-pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(pi/2) q[6];
rz(0) q[6];
sx q[6];
rz(6.188671474214624) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[2],q[6];
rz(0) q[6];
sx q[6];
rz(0.09451383296496152) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[2],q[6];
rz(-pi/2) q[2];
rz(2.0710373688121315) q[7];
rz(pi/2) q[7];
rz(1.2637318571510934) q[7];
cx q[10],q[7];
rz(-1.2637318571510934) q[7];
cx q[10],q[7];
rz(0) q[10];
sx q[10];
rz(5.207194616286485) q[10];
sx q[10];
rz(3*pi) q[10];
x q[10];
rz(-pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
sx q[7];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(-pi/4) q[8];
cx q[13],q[8];
rz(-2.4750519212854956) q[8];
cx q[13],q[8];
rz(pi/2) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[13],q[12];
rz(0.37681565461215455) q[12];
cx q[13],q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[12],q[2];
rz(2.911411940014075) q[12];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[13];
cx q[11],q[13];
cx q[13],q[11];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(5.796894385503028) q[13];
rz(-pi/2) q[13];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(0.6030722702830265) q[2];
cx q[4],q[11];
rz(-1.8491656776927452) q[11];
cx q[4],q[11];
rz(1.8491656776927452) q[11];
rz(-pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(-pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(2.4750519212854956) q[8];
cx q[8],q[1];
cx q[1],q[8];
rz(-1.9434629603546165) q[1];
cx q[3],q[1];
rz(-3.896034652853113) q[1];
sx q[1];
rz(2.4964093035195605) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[3],q[1];
rz(0) q[1];
sx q[1];
rz(3.7867760036600258) q[1];
sx q[1];
rz(15.264275573977109) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[1],q[0];
cx q[0],q[1];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[0],q[12];
rz(-4.9536144894590945) q[1];
rz(pi/2) q[1];
rz(-2.911411940014075) q[12];
cx q[0],q[12];
rz(-pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
cx q[0],q[4];
rz(pi/2) q[3];
cx q[3],q[7];
x q[3];
rz(-1.9694886035531578) q[3];
sx q[3];
rz(5.929781992258232) q[3];
sx q[3];
rz(11.394266564322537) q[3];
rz(-pi/2) q[3];
x q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(3.4822294259358695) q[4];
cx q[0],q[4];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(0) q[0];
sx q[0];
rz(9.178284736390252) q[0];
sx q[0];
rz(3*pi) q[0];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
rz(3.2642676843884413) q[4];
sx q[4];
rz(2.0697584870517973) q[4];
cx q[5],q[8];
cx q[7],q[2];
rz(-0.6030722702830265) q[2];
cx q[7],q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(0.6316329590029108) q[2];
rz(-3.931031199484959) q[7];
rz(pi/2) q[7];
cx q[11],q[7];
rz(0) q[11];
sx q[11];
rz(3.7647133814116076) q[11];
sx q[11];
rz(3*pi) q[11];
rz(0) q[7];
sx q[7];
rz(2.5184719257679786) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[11],q[7];
rz(-pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(3.403436468750318) q[11];
rz(-pi/2) q[7];
rz(3.931031199484959) q[7];
rz(2.4646656492604264) q[7];
rz(-2.146193620051242) q[8];
cx q[5],q[8];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[6],q[5];
rz(3.8041487461014207) q[5];
cx q[6],q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(-pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[5],q[1];
rz(0) q[1];
sx q[1];
rz(0.3311861283400437) q[1];
sx q[1];
rz(3*pi) q[1];
rz(0) q[5];
sx q[5];
rz(5.951999178839543) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[5],q[1];
rz(-pi/2) q[1];
rz(4.9536144894590945) q[1];
rz(2.3315917626380416) q[1];
cx q[1],q[12];
rz(-2.3315917626380416) q[12];
cx q[1],q[12];
rz(-1.5184309068055448) q[1];
cx q[11],q[1];
rz(-3.403436468750318) q[1];
sx q[1];
rz(2.1203708339473435) q[1];
sx q[1];
rz(3*pi) q[1];
cx q[11],q[1];
rz(0) q[1];
sx q[1];
rz(4.162814473232243) q[1];
sx q[1];
rz(14.346645336325242) q[1];
rz(-pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
x q[11];
rz(2.3315917626380416) q[12];
rz(pi/2) q[12];
cx q[12],q[4];
cx q[4],q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/4) q[12];
rz(2.4232558356479785) q[4];
cx q[4],q[11];
rz(-2.4232558356479785) q[11];
cx q[4],q[11];
rz(2.4232558356479785) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/4) q[4];
rz(-pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(-pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[5],q[10];
rz(4.221735023198853) q[10];
cx q[5],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(-pi/4) q[10];
rz(pi/4) q[10];
cx q[10],q[3];
rz(-pi/4) q[3];
cx q[10],q[3];
rz(5.035642065492692) q[10];
cx q[10],q[4];
rz(0.4883969224612197) q[10];
sx q[10];
rz(4.974546044480229) q[10];
sx q[10];
rz(8.93638103830816) q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
rz(3.294877728391325) q[10];
rz(pi/4) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(-pi/4) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(-pi/2) q[4];
rz(5.167386544469246) q[4];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(-pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(2.146193620051242) q[8];
rz(pi) q[8];
rz(-pi/4) q[8];
rz(0.9954373429548745) q[8];
sx q[8];
rz(6.545188952869839) q[8];
sx q[8];
rz(8.429340617814505) q[8];
rz(-pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[5];
rz(3.752086364213803) q[5];
cx q[8],q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(2.3289790444417466) q[5];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(1.0601829757088372) q[8];
cx q[8],q[7];
rz(-1.0601829757088372) q[7];
cx q[8],q[7];
rz(1.0601829757088372) q[7];
rz(-1.3852857351114503) q[7];
sx q[7];
rz(4.273835118009007) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[8],q[12];
rz(-pi/4) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(-pi/2) q[12];
x q[12];
rz(-pi/2) q[12];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(2.439216897142471) q[8];
cx q[9],q[6];
rz(3.9040671572663252) q[6];
cx q[9],q[6];
cx q[2],q[6];
rz(-0.6316329590029108) q[6];
cx q[2],q[6];
rz(1.2466382160760063) q[2];
rz(-pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[1],q[2];
rz(3.586630108902876) q[2];
cx q[1],q[2];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(2.0199470781985744) q[2];
rz(pi/2) q[2];
cx q[3],q[1];
cx q[1],q[3];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
id q[1];
rz(pi/2) q[3];
rz(4.808293878656198) q[3];
rz(0) q[3];
sx q[3];
rz(8.587793199995001) q[3];
sx q[3];
rz(3*pi) q[3];
rz(pi/2) q[3];
rz(0.6316329590029108) q[6];
cx q[9],q[13];
rz(pi/2) q[13];
sx q[13];
rz(4.209155743974537) q[13];
rz(2.738893731783403) q[13];
cx q[13],q[5];
rz(-2.738893731783403) q[5];
sx q[5];
rz(2.2333519743442967) q[5];
sx q[5];
rz(3*pi) q[5];
cx q[13],q[5];
rz(pi/2) q[13];
sx q[13];
rz(4.720354509969555) q[13];
sx q[13];
rz(5*pi/2) q[13];
rz(0) q[5];
sx q[5];
rz(4.04983333283529) q[5];
sx q[5];
rz(9.834692648111035) q[5];
x q[5];
cx q[5],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/4) q[5];
cx q[5],q[11];
rz(-pi/4) q[11];
cx q[5],q[11];
rz(pi/4) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
rz(-pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[12],q[5];
rz(2.6455347825579354) q[5];
cx q[12],q[5];
rz(pi/2) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/2) q[12];
rz(0) q[12];
sx q[12];
rz(5.241362201850983) q[12];
sx q[12];
rz(3*pi) q[12];
rz(1.172702845316925) q[12];
sx q[12];
rz(2.8496167734655695) q[12];
rz(-0.7405537323288491) q[12];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
cx q[6],q[9];
cx q[9],q[6];
rz(-4.495444252037694) q[6];
rz(pi/2) q[6];
cx q[0],q[6];
rz(0) q[0];
sx q[0];
rz(3.3185767300476545) q[0];
sx q[0];
rz(3*pi) q[0];
rz(0) q[6];
sx q[6];
rz(2.9646085771319317) q[6];
sx q[6];
rz(3*pi) q[6];
cx q[0],q[6];
rz(-pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
rz(5.91348226802306) q[0];
cx q[0],q[8];
rz(-pi/2) q[6];
rz(4.495444252037694) q[6];
rz(0.8144559681335252) q[6];
cx q[13],q[6];
cx q[6],q[13];
cx q[13],q[6];
rz(-0.5148866263848513) q[13];
cx q[4],q[13];
rz(-5.167386544469246) q[13];
sx q[13];
rz(0.6383497559134157) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[4],q[13];
rz(0) q[13];
sx q[13];
rz(5.644835551266171) q[13];
sx q[13];
rz(15.107051131623477) q[13];
rz(0) q[4];
sx q[4];
rz(3.656546660851709) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[5],q[13];
rz(0.9339508407257104) q[13];
cx q[5],q[13];
rz(-2.4177662431791003) q[13];
rz(-pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[1],q[6];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
id q[6];
rz(5.5913699239690136) q[6];
rz(3.3475858428349694) q[6];
cx q[6],q[13];
rz(-3.3475858428349694) q[13];
sx q[13];
rz(3.0503519461932256) q[13];
sx q[13];
rz(3*pi) q[13];
cx q[6],q[13];
rz(0) q[13];
sx q[13];
rz(3.2328333609863606) q[13];
sx q[13];
rz(15.190130046783448) q[13];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(0.497517087897336) q[13];
cx q[10],q[13];
rz(-0.497517087897336) q[13];
cx q[10],q[13];
rz(pi/2) q[10];
sx q[10];
rz(7.603363137059721) q[10];
sx q[10];
rz(5*pi/2) q[10];
rz(pi/2) q[10];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(-1.608350340903606) q[13];
sx q[13];
rz(5.7954367596807135) q[13];
sx q[13];
rz(11.033128301672985) q[13];
rz(-pi/2) q[6];
rz(-2.439216897142471) q[8];
cx q[0],q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(-pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(-pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[2];
rz(0) q[2];
sx q[2];
rz(0.0623642243263518) q[2];
sx q[2];
rz(3*pi) q[2];
rz(0) q[9];
sx q[9];
rz(0.0623642243263518) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[9],q[2];
rz(-pi/2) q[2];
rz(-2.0199470781985744) q[2];
id q[2];
cx q[0],q[2];
cx q[2],q[0];
cx q[0],q[2];
rz(pi/4) q[0];
rz(2.6538209648694986) q[0];
rz(pi/4) q[2];
cx q[2],q[1];
rz(-pi/4) q[1];
cx q[2],q[1];
rz(pi/4) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(0) q[1];
sx q[1];
rz(4.982694315812725) q[1];
sx q[1];
rz(3*pi) q[1];
id q[1];
rz(-0.880328870249691) q[1];
rz(pi/2) q[1];
rz(pi) q[2];
rz(-pi/4) q[2];
rz(-pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[2],q[1];
rz(0) q[1];
sx q[1];
rz(1.7882322630826748) q[1];
sx q[1];
rz(3*pi) q[1];
rz(0) q[2];
sx q[2];
rz(4.494953044096912) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[2],q[1];
rz(-pi/2) q[1];
rz(0.880328870249691) q[1];
rz(-pi/2) q[1];
rz(-pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
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
cx q[9],q[7];
rz(1.3173416106272493) q[7];
cx q[9],q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(-pi/2) q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[8];
rz(0.29523156965677033) q[8];
cx q[7],q[8];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[7];
id q[7];
cx q[7],q[0];
rz(-2.6538209648694986) q[0];
cx q[7],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
cx q[8],q[4];
rz(0) q[4];
sx q[4];
rz(2.6266386463278772) q[4];
sx q[4];
rz(3*pi) q[4];
cx q[8],q[4];
rz(pi/2) q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[4],q[3];
cx q[3],q[4];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(4.500196897001937) q[3];
cx q[3],q[12];
rz(-4.500196897001937) q[12];
sx q[12];
rz(1.1648755132206856) q[12];
sx q[12];
rz(3*pi) q[12];
cx q[3],q[12];
rz(0) q[12];
sx q[12];
rz(5.1183097939589) q[12];
sx q[12];
rz(14.665528590100166) q[12];
rz(pi/2) q[12];
sx q[12];
rz(6.669075536399017) q[12];
sx q[12];
rz(5*pi/2) q[12];
rz(-pi/2) q[12];
rz(-pi/4) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[3],q[2];
rz(2.903084996730421) q[2];
cx q[3],q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/4) q[4];
cx q[4],q[0];
rz(-pi/4) q[0];
cx q[4],q[0];
rz(pi/4) q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi/2) q[0];
rz(pi/4) q[4];
cx q[6],q[0];
rz(pi/2) q[0];
rz(pi/2) q[0];
sx q[0];
rz(6.684185774251682) q[0];
sx q[0];
rz(5*pi/2) q[0];
rz(pi/2) q[0];
rz(2.0870276437209756) q[0];
rz(pi/2) q[0];
rz(3.222389816357813) q[6];
rz(-pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[5];
rz(4.953372923890243) q[5];
cx q[8],q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(-pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
rz(-4.030319156635732) q[8];
rz(pi/2) q[8];
cx q[5],q[8];
rz(0) q[5];
sx q[5];
rz(4.249861476641448) q[5];
sx q[5];
rz(3*pi) q[5];
rz(0) q[8];
sx q[8];
rz(2.033323830538138) q[8];
sx q[8];
rz(3*pi) q[8];
cx q[5],q[8];
rz(-pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(pi/2) q[5];
rz(-pi/2) q[8];
rz(4.030319156635732) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[4],q[8];
rz(-pi/4) q[8];
cx q[4],q[8];
cx q[4],q[12];
rz(pi/2) q[12];
rz(1.3108153402728544) q[12];
rz(pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(0.6516212355918868) q[8];
cx q[8],q[13];
rz(-0.6516212355918868) q[13];
cx q[8],q[13];
rz(0.6516212355918868) q[13];
rz(pi) q[13];
rz(-1.2586492857152054) q[13];
rz(pi/2) q[13];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[3],q[8];
rz(pi/4) q[3];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[3],q[8];
rz(-pi/4) q[8];
cx q[3],q[8];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(-pi/4) q[9];
cx q[11],q[9];
cx q[9],q[11];
cx q[11],q[9];
rz(0.8298783906610823) q[9];
cx q[9],q[11];
rz(-0.8298783906610823) q[11];
cx q[9],q[11];
rz(0.8298783906610823) q[11];
cx q[11],q[7];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/4) q[11];
cx q[7],q[11];
rz(-pi/4) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
rz(-pi/2) q[11];
rz(pi/2) q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[11],q[10];
cx q[10],q[11];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
rz(2.8570272804510073) q[10];
id q[11];
rz(pi/4) q[11];
cx q[11],q[8];
cx q[2],q[10];
rz(-2.8570272804510073) q[10];
cx q[2],q[10];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
sx q[10];
rz(-pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[2],q[13];
rz(0) q[13];
sx q[13];
rz(0.13611790516681443) q[13];
sx q[13];
rz(3*pi) q[13];
rz(0) q[2];
sx q[2];
rz(6.147067402012771) q[2];
sx q[2];
rz(3*pi) q[2];
cx q[2],q[13];
rz(-pi/2) q[13];
rz(1.2586492857152054) q[13];
rz(-pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(pi/2) q[2];
rz(-2.3691763801913757) q[7];
cx q[6],q[7];
rz(-3.222389816357813) q[7];
sx q[7];
rz(0.6202823616913196) q[7];
sx q[7];
rz(3*pi) q[7];
cx q[6],q[7];
cx q[4],q[6];
cx q[6],q[4];
rz(pi/2) q[4];
cx q[4],q[10];
x q[4];
rz(pi/4) q[6];
cx q[6],q[3];
rz(-pi/4) q[3];
cx q[6],q[3];
rz(pi/4) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(0) q[7];
sx q[7];
rz(5.662902945488266) q[7];
sx q[7];
rz(15.016344157318567) q[7];
cx q[12],q[7];
rz(-1.3108153402728544) q[7];
cx q[12],q[7];
rz(1.3108153402728544) q[7];
rz(0.9932419226774591) q[7];
sx q[7];
rz(2.5377100923044633) q[7];
rz(-pi/4) q[8];
cx q[11],q[8];
rz(pi/4) q[8];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
rz(2.8870900550008822) q[9];
rz(0) q[9];
sx q[9];
rz(3.664091635585325) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[5],q[9];
rz(0) q[9];
sx q[9];
rz(2.6190936715942614) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[5],q[9];
cx q[5],q[1];
rz(pi/2) q[1];
rz(0.1677366350079392) q[1];
cx q[1],q[12];
rz(3.06489069931847) q[5];
sx q[5];
rz(6.223572933396182) q[5];
rz(pi/4) q[5];
x q[9];
rz(5.747844193057577) q[9];
rz(-pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[9],q[0];
rz(0) q[0];
sx q[0];
rz(2.0827732568866626) q[0];
sx q[0];
rz(3*pi) q[0];
rz(0) q[9];
sx q[9];
rz(2.0827732568866626) q[9];
sx q[9];
rz(3*pi) q[9];
cx q[9],q[0];
rz(-pi/2) q[0];
rz(-2.0870276437209756) q[0];
rz(-pi/2) q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[9];
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
