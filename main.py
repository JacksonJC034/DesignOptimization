#!/usr/bin/env python3
"""URDF fixed-base kinematics and inverse dynamics verification.

Install: python3 -m pip install numpy
Run:     python3 main.py --self-test
         python3 main.py --init-config verification_config.json
         python3 main.py --config verification_config.json
         python3 main.py --demo-motion --output verification_demo

All calculations use SI units and the URDF root frame (fixed in space).
URDF inertias are about each link COM in the inertial frame, including inertial
origin rotation. Payloads are point masses, specified once at a link-local COM.
No friction, rotor inertia, gearbox losses, contact solver or mesh collision.
The default zero-pose hold is a smoke test, NOT a validated sit/stand motion.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_URDF = HERE / 'URDF+STP/VC-urdf-260623/urdf/VC-urdf-260623.urdf'


def vec(value, n=3):
    a = np.asarray(value.split() if isinstance(value, str) else value, dtype=float)
    if a.shape != (n,) or not np.isfinite(a).all():
        raise ValueError(f'Expected {n} finite numbers, got {value!r}')
    return a


def rotation(axis, angle):
    a = vec(axis)
    a = a / np.linalg.norm(a)
    x, y, z = a
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle)*k + (1-math.cos(angle))*(k@k)


def origin(element):
    if element is None:
        return np.zeros(3), np.eye(3)
    p = vec(element.get('xyz', '0 0 0'))
    r, pitch, y = vec(element.get('rpy', '0 0 0'))
    return p, rotation([0, 0, 1], y) @ rotation([0, 1, 0], pitch) @ rotation([1, 0, 0], r)


class Robot:
    """Tree URDF model; world-frame recursive Newton-Euler implementation."""
    def __init__(self, path):
        self.path = Path(path)
        root = ET.parse(path).getroot()
        self.links, self.joints, self.warnings = {}, [], []
        for e in root.findall('link'):
            name = e.get('name')
            if name in self.links:
                raise ValueError(f'Duplicate link {name}')
            it = e.find('inertial')
            if it is None:
                raise ValueError(f'{name}: missing inertial data; add explicit mass/inertia')
            c, ri = origin(it.find('origin'))
            mass = float(it.find('mass').get('value'))
            ie = it.find('inertia')
            xx, yy, zz, xy, xz, yz = [float(ie.get(k)) for k in ('ixx','iyy','izz','ixy','ixz','iyz')]
            inertia = np.array([[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]])
            eig = np.linalg.eigvalsh(inertia)
            if not np.isfinite(mass) or mass < 0 or not np.isfinite(inertia).all() or min(eig) < -1e-10:
                raise ValueError(f'{name}: invalid mass/inertia')
            if eig[-1] > eig[0]+eig[1]+1e-8:
                raise ValueError(f'{name}: inertia violates principal-moment triangle inequality')
            if mass == 0 and np.linalg.norm(inertia) > 1e-10:
                raise ValueError(f'{name}: nonzero inertia on zero-mass link')
            self.links[name] = dict(m=mass, c=c, I=ri@inertia@ri.T)
        pending, children, names = [], set(), set()
        for e in root.findall('joint'):
            name, kind = e.get('name'), e.get('type')
            if name in names:
                raise ValueError(f'Duplicate joint {name}')
            names.add(name)
            if kind not in ('fixed','revolute','continuous','prismatic') or e.find('mimic') is not None:
                raise ValueError(f'{name}: unsupported joint type or mimic relation')
            parent, child = e.find('parent').get('link'), e.find('child').get('link')
            if parent not in self.links or child not in self.links or child in children:
                raise ValueError(f'{name}: invalid parent/child or multiple parents')
            children.add(child)
            p, r = origin(e.find('origin'))
            ae = e.find('axis')
            axis = vec(ae.get('xyz', '1 0 0') if ae is not None else '1 0 0')
            if np.linalg.norm(axis) < 1e-12:
                raise ValueError(f'{name}: zero joint axis')
            limit = e.find('limit')
            limits = {} if limit is None else {k: float(v) for k,v in limit.attrib.items()}
            if not all(math.isfinite(v) for v in limits.values()):
                raise ValueError(f'{name}: nonfinite limits')
            if kind in ('revolute','prismatic') and not ('lower' in limits and 'upper' in limits and limits['lower'] <= limits['upper']):
                raise ValueError(f'{name}: missing or inverted position limits')
            pending.append(dict(name=name, kind=kind, parent=parent, child=child,
                                p=p, R=r, axis=axis/np.linalg.norm(axis), limits=limits))
        roots = set(self.links)-children
        if len(roots) != 1:
            raise ValueError('URDF must have exactly one root')
        self.root = roots.pop()
        visited = {self.root}
        while pending:
            batch = [j for j in pending if j['parent'] in visited]
            if not batch:
                raise ValueError('Disconnected or cyclic URDF')
            for j in batch:
                self.joints.append(j)
                visited.add(j['child'])
                pending.remove(j)
        if visited != set(self.links):
            raise ValueError('Disconnected links')
        self.active = [j for j in self.joints if j['kind'] != 'fixed']
        self.names = [j['name'] for j in self.active]
        self.index = {n:i for i,n in enumerate(self.names)}
        for j in self.active:
            for key in ('effort','velocity'):
                if j['limits'].get(key, 0) <= 0:
                    self.warnings.append(f"{j['name']}: {key} missing/nonpositive; capacity check unavailable")
        self.mass = sum(l['m'] for l in self.links.values())
        if self.mass <= 0:
            raise ValueError('Total model mass must be positive')

    def state(self, q, qd=None, qdd=None):
        n = len(self.active)
        q = vec(q, n)
        qd = np.zeros(n) if qd is None else vec(qd, n)
        qdd = np.zeros(n) if qdd is None else vec(qdd, n)
        # p, R, angular velocity, angular acceleration, origin acceleration
        z = np.zeros(3)
        states = {self.root: (z.copy(), np.eye(3), z.copy(), z.copy(), z.copy())}
        axes = {}
        for j in self.joints:
            p, r, w, alpha, acc = states[j['parent']]
            r0 = r@j['R']
            a = r0@j['axis']
            displacement = r@j['p']
            kind = j['kind']
            v = vd = vdd = 0.
            if kind != 'fixed':
                i = self.index[j['name']]
                v, vd, vdd = q[i], qd[i], qdd[i]
            if kind == 'prismatic':
                displacement = displacement + a*v
            pc = p + displacement
            ac = acc + np.cross(alpha, displacement) + np.cross(w, np.cross(w, displacement))
            wc, alphac, rc = w.copy(), alpha.copy(), r0
            if kind in ('revolute','continuous'):
                rc = r0@rotation(j['axis'], v)
                wc = w+a*vd
                alphac = alpha+a*vdd+np.cross(w, a*vd)
            elif kind == 'prismatic':
                ac = ac+a*vdd+2*np.cross(w, a*vd)
            states[j['child']] = (pc, rc, wc, alphac, ac)
            axes[j['name']] = a
        return states, axes

    def evaluate(self, q, qd=None, qdd=None, payloads=(), gravity=(0,0,-9.80665)):
        s, axes = self.state(q, qd, qdd)
        g = vec(gravity)
        forces, moments, com_sum = {}, {}, np.zeros(3)
        mass = self.mass
        def add_body(link, m, local_com, inertia):
            nonlocal com_sum
            p, r, w, alpha, acc = s[link]
            offset = r@local_com
            cg = p+offset
            ac = acc+np.cross(alpha, offset)+np.cross(w, np.cross(w, offset))
            iw = r@inertia@r.T
            f = m*(ac-g)
            moment = iw@alpha+np.cross(w, iw@w)+np.cross(offset, f)
            forces[link] = forces.get(link, np.zeros(3))+f
            moments[link] = moments.get(link, np.zeros(3))+moment
            com_sum += m*cg
        for name, link in self.links.items():
            add_body(name, link['m'], link['c'], link['I'])
        for payload in payloads:
            if payload['link'] not in self.links:
                raise ValueError(f"Unknown payload link: {payload['link']}")
            m = float(payload['mass_kg'])
            if not math.isfinite(m) or m < 0:
                raise ValueError('Payload mass must be finite and nonnegative')
            add_body(payload['link'], m, vec(payload.get('com_xyz_m',[0,0,0])), np.zeros((3,3)))
            mass += m
        tau = np.zeros(len(self.active))
        for j in reversed(self.joints):
            child, parent = j['child'], j['parent']
            if j['kind'] != 'fixed':
                wrench = forces[child] if j['kind'] == 'prismatic' else moments[child]
                tau[self.index[j['name']]] = axes[j['name']]@wrench
            forces[parent] += forces[child]
            moments[parent] += moments[child]+np.cross(s[child][0]-s[parent][0], forces[child])
        return dict(tau=tau, com=com_sum/mass, mass=mass, states=s,
                    base_force=forces[self.root], base_moment=moments[self.root])


def config_template(robot):
    zero = {j['name']: float(np.clip(0, j['limits'].get('lower', -np.inf), j['limits'].get('upper', np.inf)))
            if j['kind'] != 'continuous' else 0. for j in robot.active}
    return dict(case_name='neutral_hold_not_a_sit_stand_validation', duration_s=4., samples=201,
                start_q=zero, end_q=zero.copy(), gravity_m_s2=[0,0,-9.80665], payloads=[],
                monitor_links=['torso','left_arm_link_7','right_arm_link_7'],
                support_polygon_xy_m=None, support_margin_m=0.,
                capacity_overrides={},
                notes=['Angles in radians; prismatic joints in metres.',
                       'Set real sit/stand start_q and end_q. All joints must be specified.',
                       'payloads: [{"link":"left_arm_link_7","mass_kg":3,"com_xyz_m":[0,0,0]}]',
                       'Payload COM is relative to link frame, not an assumed tool frame.',
                       'Support polygon: actual convex contact region in root XY, CCW or CW.',
                       'Do not infer a support polygon from wheel joint centres.',
                       'capacity_overrides: joint -> {effort: positive number, velocity: positive number}.',
                       'Overrides are constant screening bounds, not full torque-speed/thermal curves.'])


def polygon_margin(point, polygon):
    p = np.asarray(polygon, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all():
        raise ValueError('Support polygon requires >=3 finite XY vertices')
    edge = np.roll(p,-1,axis=0)-p
    if np.any(np.linalg.norm(edge,axis=1) < 1e-10):
        raise ValueError('Repeated support polygon vertices')
    area2 = np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1))
    if abs(area2) < 1e-10:
        raise ValueError('Support polygon is degenerate')
    if area2 < 0:
        p = p[::-1]; edge = np.roll(p,-1,axis=0)-p
    # All vertices must be on the inner side of every edge: convex, ordered polygon.
    for a,e in zip(p,edge):
        v = p-a
        if np.any(e[0]*v[:,1]-e[1]*v[:,0] < -1e-10):
            raise ValueError('Support polygon must be convex and boundary-ordered')
    v = np.asarray(point)-p
    return float(np.min((edge[:,0]*v[:,1]-edge[:,1]*v[:,0])/np.linalg.norm(edge,axis=1)))


def make_trajectory(robot, cfg):
    for key in ('start_q','end_q'):
        if set(cfg[key]) != set(robot.names):
            raise ValueError(f'{key} must specify every active joint exactly once')
    duration = float(cfg['duration_s']); count = cfg['samples']
    if not math.isfinite(duration) or duration <= 0 or isinstance(count,bool) or not isinstance(count,int) or count < 3:
        raise ValueError('duration_s must be positive; samples must be integer >=3')
    t = np.linspace(0,duration,count); u = t/duration
    # Quintic rest-to-rest interpolation, analytic derivatives (no numerical differentiation).
    s = 10*u**3-15*u**4+6*u**5
    sd = (30*u**2-60*u**3+30*u**4)/duration
    sdd = (60*u-180*u**2+120*u**3)/duration**2
    a = vec([cfg['start_q'][n] for n in robot.names],len(robot.names))
    b = vec([cfg['end_q'][n] for n in robot.names],len(robot.names))
    return t, a+s[:,None]*(b-a), sd[:,None]*(b-a), sdd[:,None]*(b-a)


def write_csv(path, headers, rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(headers);w.writerows(rows)


def chart(title, t, data, labels):
    """Small offline SVG plot, embedded in HTML; raw values also exported as CSV."""
    data=np.asarray(data); data=data[:,None] if data.ndim==1 else data
    lo,hi=float(data.min()),float(data.max()); pad=max((hi-lo)*.08,1e-6)
    lo-=pad;hi+=pad
    colors=['#1769aa','#d1495b','#238b45','#8c49a3']
    parts=[f'<h3>{html.escape(title)}</h3><svg viewBox="0 0 920 310" role="img" aria-label="{html.escape(title)}">']
    for k in range(5):
        y=20+k*55; value=hi-(hi-lo)*k/4
        parts.append(f'<path d="M80 {y}H890" stroke="#ddd"/><text x="3" y="{y+5}" font-size="14">{value:.4g}</text>')
    for k in range(data.shape[1]):
        coords=' '.join(f'{80+810*tt/t[-1]:.2f},{20+220*(hi-v)/(hi-lo):.2f}' for tt,v in zip(t,data[:,k]))
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{colors[k%4]}" stroke-width="2"/>')
    parts.append(f'<text x="80" y="270">0 s</text><text x="825" y="270">{t[-1]:.3g} s</text></svg>')
    parts.append('<p>'+ ' | '.join(f'<span style="color:{colors[k%4]}">{html.escape(v)}</span>' for k,v in enumerate(labels))+'</p>')
    return ''.join(parts)


def write_visualizations(robot, cfg, out, t, q, qd, qdd, tau, com, results, joint_report):
    """Offline 3D skeleton animation, time curves, limit bars and SVG exports."""
    import re
    names = list(robot.links)
    index = {n:i for i,n in enumerate(names)}
    frames = np.array([[r['states'][n][0] for n in names] for r in results])
    payload_frames = []
    for payload in cfg['payloads']:
        offset = vec(payload.get('com_xyz_m',[0,0,0]))
        payload_frames.append([(r['states'][payload['link']][0]+r['states'][payload['link']][1]@offset).tolist() for r in results])
    data = dict(time=t.tolist(),frames=frames.tolist(),com=com.tolist(),
                edges=[[index[j['parent']],index[j['child']]] for j in robot.joints],
                links=names,monitor=[index[n] for n in cfg['monitor_links']],payloads=payload_frames,
                joints=robot.names,q=q.tolist(),qd=qd.tolist(),qdd=qdd.tolist(),tau=tau.tolist(),
                summary=joint_report,position_limits=[None if j['kind']=='continuous' else [j['limits']['lower'],j['limits']['upper']] for j in robot.active],
                units=['m' if j['kind']=='prismatic' else 'rad' for j in robot.active],
                polygon=cfg['support_polygon_xy_m'],case=str(cfg['case_name']),
                stationary=bool(np.allclose(q[0],q[-1])))
    # Root-relative positions; skeleton edges connect frame origins, not CAD geometry.
    graphs=out/'figures';graphs.mkdir(exist_ok=True)
    def export_svg(stem,title,values,labels):
        markup=chart(title,t,values,labels)
        svg=re.search(r'<svg.*?</svg>',markup,re.S).group(0)
        svg=svg.replace('0 0 920 310','0 0 920 360').replace('</svg>', '<text x="80" y="315" font-size="18">'+html.escape(title)+'</text><text x="80" y="343" font-size="14">'+html.escape(' | '.join(labels))+'</text></svg>')
        svg=svg.replace('<svg ','<svg xmlns="http://www.w3.org/2000/svg" ',1)
        (graphs/(stem+'.svg')).write_text(svg,encoding='utf-8')
    export_svg('com','COM position (m)',com,['x','y','z'])
    for i,j in enumerate(robot.active):
        # Numeric prefixes avoid accepting a URDF name as an output path.
        export_svg(f'joint_{i:02d}_position',j['name']+' position ('+data['units'][i]+')',q[:,i],['position'])
        export_svg(f'joint_{i:02d}_effort',j['name']+' effort ('+joint_report[i]['effort_unit']+')',tau[:,i],['effort'])
    (graphs/'index.json').write_text(json.dumps(dict(joints=robot.names,description='joint_NN_* uses zero-based order in joints; SVG plots use seconds on X'),indent=2),encoding='utf-8')
    # Static spatial trajectory overview, two orthographic projections in metres.
    parts=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 560"><rect width="1100" height="560" fill="white"/>']
    colors=['#1769aa','#d1495b','#238b45','#8c49a3']
    series=[com]+[frames[:,i] for i in data['monitor']]
    labels=['COM']+[names[i] for i in data['monitor']]
    for panel,(a,b,title) in enumerate([(0,2,'XZ side projection (m)'),(0,1,'XY top projection (m)')]):
        allpoints=np.concatenate(series);low=allpoints[:,[a,b]].min(axis=0);high=allpoints[:,[a,b]].max(axis=0)
        center=(low+high)/2;span=max(float((high-low).max())*1.2,.2);scale=400/span
        def xy(v):return 275+550*panel+(v[a]-center[0])*scale,270-(v[b]-center[1])*scale
        parts.append(f'<text x="{50+550*panel}" y="28" font-size="19">{title}</text>')
        for k,values in enumerate(series):
            coords=' '.join(f'{xy(v)[0]:.2f},{xy(v)[1]:.2f}' for v in values)
            sx,sy=xy(values[0]);ex,ey=xy(values[-1]);c=colors[k%4]
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{c}" stroke-width="2"/><circle cx="{sx}" cy="{sy}" r="5" fill="white" stroke="{c}"/><circle cx="{ex}" cy="{ey}" r="4" fill="{c}"/><text x="{35+550*panel}" y="{475+k*22}" font-size="14" fill="{c}">{html.escape(labels[k])}</text>')
        # Physical scale bar, with equal X/Y scale in each projection.
        length=span/5
        parts.append(f'<path d="M{60+550*panel} 440h80" stroke="black"/><text x="{60+550*panel}" y="430" font-size="14">{length:.3g} m</text>')
    parts.append('</svg>');(graphs/'spatial_trajectories.svg').write_text(''.join(parts),encoding='utf-8')
    template=r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>整机运动与验证结果</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f5f7fa;color:#1b2d40;font:15px system-ui,-apple-system,sans-serif}main{max-width:1400px;margin:auto;padding:24px}h1{font-size:27px;margin:0 0 10px}h2{font-size:20px;margin:0 0 15px}.muted{color:#59697b}section{background:white;border:1px solid #dbe2ea;border-radius:10px;padding:20px;margin:18px 0}.row{display:flex;gap:14px;align-items:center;flex-wrap:wrap}button,select,input{font:inherit}button,select{border:1px solid #bac7d5;border-radius:5px;padding:8px;background:white;color:#1b2d40}button{cursor:pointer}button:hover{background:#edf4ff}input[type=range]{flex:1;min-width:180px}canvas{display:block;width:100%;height:540px;background:#fafcff;border:1px solid #e5ebf2;margin:14px 0;touch-action:none}.cols{display:grid;grid-template-columns:1fr 1fr;gap:15px}.plot{width:100%;height:auto}.legend span{margin-right:15px}.badge{padding:4px 8px;background:#fff0cc;border-radius:4px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;padding:9px;border-bottom:1px solid #e2e7ee;white-space:nowrap}th{position:sticky;top:0;background:#f4f7fb}.table{overflow:auto;max-height:540px}.pass{color:#177546}.fail{color:#bd2639;font-weight:600}.unknown{color:#89691b}a{color:#1769aa}.metric{min-width:160px}.metric b{display:block;font-size:22px}svg text{font-family:system-ui}details{margin:12px 0}.note{line-height:1.7} @media(max-width:800px){.cols{grid-template-columns:1fr}main{padding:12px}canvas{height:420px}}
</style><main><h1>整机运动与验证结果</h1><p id="case" class="muted"></p><p class="note">固定底盘模型 · 连杆坐标系骨架示意 · <span class="badge">整机设计尚未完成验证</span></p>
<section><h2>轨迹回放与空间路径</h2><p id="motionNote" class="note"></p><div class="row"><button id="play">播放</button><button id="reset">复位视角</button><label>视图 <select id="view"><option value="3d">三维</option><option value="xz">XZ 侧视</option><option value="yz">YZ 正视</option><option value="xy">XY 俯视</option></select></label><label>速度 <select id="speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label><label><input id="trails" type="checkbox" checked>显示完整轨迹</label><label><input id="labels" type="checkbox">显示连杆名称</label></div>
<canvas id="scene" aria-label="可旋转的机器人骨架和轨迹"></canvas><div class="row"><input id="seek" type="range" min="0" value="0" step="1" aria-label="轨迹时间"><output id="time"></output></div><p class="muted">三维视图拖动旋转、滚轮缩放。线段连接 URDF 坐标系原点，不代表连杆外形；彩色路径为监测连杆原点，橙点为整机质心，紫点为附加载荷质心。所有坐标相对固定根坐标系，单位 m。</p><div id="legend" class="legend"></div><div class="row" id="readout"></div><p><a href="figures/spatial_trajectories.svg">导出空间轨迹 SVG</a> · <a href="body_timeseries.csv">位姿及质心数据 CSV</a></p></section>
<section><h2>关节时序与模型限值</h2><div class="row"><label>关节 <select id="joint"></select></label><span id="jointState"></span></div><div class="cols"><div id="position"></div><div id="velocity"></div><div id="acceleration"></div><div id="effort"></div></div><p class="muted">灰色虚线为模型限值或配置覆盖值，橙色竖线对应回放时间。转动关节使用 rad、rad/s、rad/s² 和 N·m；移动关节使用 m、m/s、m/s² 和 N。力矩限值为初筛边界，不代替模组转矩转速曲线或热校核。</p><p><a id="positionSvg">导出当前关节位移 SVG</a> · <a id="effortSvg">导出当前关节力矩 SVG</a> · <a href="joint_timeseries.csv">全部关节时序 CSV</a></p></section>
<section><h2>验证结果总览</h2><p class="muted">能力占用率 = 轨迹最大绝对需求 ÷ 正值限值。限值缺失显示“未评估”；100% 以上为超限。</p><div class="table"><table><thead><tr><th>关节</th><th>位置限位</th><th>速度检查</th><th>速度占用</th><th>力矩或力检查</th><th>力矩或力占用</th><th>峰值需求</th><th>RMS 需求</th></tr></thead><tbody id="summary"></tbody></table></div><p><a href="joint_summary.csv">验证汇总 CSV</a> · <a href="report.json">完整结果 JSON</a> · <a href="report.html">计算范围与原始报告</a></p></section>
<section><h2>质心与支撑区域</h2><div id="comPlot"></div><p><span style="color:#1769aa">X</span> · <span style="color:#d1495b">Y</span> · <span style="color:#238b45">Z</span></p><p id="supportNote" class="note"></p><p class="muted">质心投影仅用于准静态几何筛查，动态稳定性、实际接触力和碰撞仍需进一步验证。</p></section></main>
<script id="data" type="application/json">__DATA__</script><script>
'use strict';
const D=JSON.parse(document.getElementById('data').textContent),el=id=>document.getElementById(id),C=['#1769aa','#d1495b','#238b45','#8c49a3'];
let frame=0,playing=false,last=null,clockTime=0,yaw=-.65,pitch=.35,zoom=1,drag=null;
el('case').textContent='工况：'+D.case+' ｜ '+D.frames[0].length+' 个连杆 ｜ '+D.time.length+' 个采样点';
el('motionNote').textContent=D.stationary?'当前是零位或固定姿态保持，起止关节角相同，播放时骨架不会运动。请在配置中填写真实起止姿态后重新运行。':'当前展示配置指定的五次多项式关节插值；它不是经过碰撞和接触验证的坐站轨迹。';
el('seek').max=D.time.length-1;
D.monitor.forEach((i,k)=>{let s=document.createElement('span');s.style.color=C[k%4];s.textContent=D.links[i];el('legend').appendChild(s)});
D.joints.forEach((name,i)=>{let o=document.createElement('option');o.value=i;o.textContent=name;el('joint').appendChild(o)});
const status=v=>v===null?['未评估','unknown']:v?['通过','pass']:['超限','fail'];
D.summary.forEach(r=>{let tr=document.createElement('tr');const value=(v)=>{let td=document.createElement('td');td.textContent=v;tr.appendChild(td);return td};value(r.joint);['position_pass','velocity_pass'].forEach(k=>{let [s,c]=status(r[k]);value(s).className=c});value(r.velocity_limit? (100*r.peak_abs_velocity/r.velocity_limit).toFixed(1)+'%':'—');let[s,c]=status(r.effort_pass);value(s).className=c;value(r.effort_limit?(100*r.peak_abs_effort/r.effort_limit).toFixed(1)+'%':'—');value(r.peak_abs_effort.toFixed(3)+' '+r.effort_unit);value(r.rms_effort.toFixed(3)+' '+r.effort_unit);el('summary').appendChild(tr)});
const all=D.frames.flat().concat(D.com),low=[0,1,2].map(k=>Math.min(...all.map(p=>p[k]))),high=[0,1,2].map(k=>Math.max(...all.map(p=>p[k]))),center=low.map((v,k)=>(v+high[k])/2),span=Math.max(...high.map((v,k)=>v-low[k]),.3);
const canvas=el('scene'),ctx=canvas.getContext('2d');
function project(p,W,H){let[x,y,z]=p.map((v,k)=>v-center[k]),a,b,view=el('view').value;if(view==='xy'){a=x;b=y}else if(view==='xz'){a=x;b=z}else if(view==='yz'){a=y;b=z}else{let xx=Math.cos(yaw)*x-Math.sin(yaw)*y,yy=Math.sin(yaw)*x+Math.cos(yaw)*y;a=xx;b=Math.cos(pitch)*z-Math.sin(pitch)*yy}let scale=Math.min(W,H)*.68/span*zoom;return [W/2+a*scale,H/2-b*scale]}
function draw(){let W=canvas.clientWidth,H=canvas.clientHeight,ratio=window.devicePixelRatio||1;if(canvas.width!==Math.round(W*ratio)||canvas.height!==Math.round(H*ratio)){canvas.width=Math.round(W*ratio);canvas.height=Math.round(H*ratio)}ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,W,H);
function line(points,color,width=2,dash=[]){ctx.beginPath();points.forEach((p,i)=>{let[x,y]=project(p,W,H);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.stroke();ctx.setLineDash([])}
function dot(p,color,r=5){let[x,y]=project(p,W,H);ctx.beginPath();ctx.arc(x,y,r,0,2*Math.PI);ctx.fillStyle=color;ctx.fill()}
let ground=low[2]-.02,extent=Math.max(span*.7,.3);for(let k=-5;k<=5;k++){let v=k*extent/5;line([[-extent,v,ground],[extent,v,ground]],'#e2e8f0',1);line([[v,-extent,ground],[v,extent,ground]],'#e2e8f0',1)}
if(D.polygon){line(D.polygon.concat([D.polygon[0]]).map(p=>[p[0],p[1],ground]),'#177546',3);let p=D.com[frame];dot([p[0],p[1],ground],'#e88a17',5);line([p,[p[0],p[1],ground]],'#e88a17',1,[4,4])}
if(el('trails').checked){D.monitor.forEach((n,k)=>line(D.frames.map(f=>f[n]),C[k%4],2));line(D.com,'#e88a17',2,[5,3])}
D.edges.forEach(([a,b])=>line([D.frames[frame][a],D.frames[frame][b]],'#718398',4));D.frames[frame].forEach(p=>dot(p,'#2c4058',3));D.monitor.forEach((n,k)=>dot(D.frames[frame][n],C[k%4],6));dot(D.com[frame],'#e88a17',8);D.payloads.forEach(p=>dot(p[frame],'#8c49a3',7));
ctx.font='12px system-ui';ctx.fillStyle='#34495e';if(el('labels').checked)D.frames[frame].forEach((p,i)=>{let[x,y]=project(p,W,H);ctx.fillText(D.links[i],x+7,y-5)});
[['X',[.2,0,0],'#c43c4d'],['Y',[0,.2,0],'#27834d'],['Z',[0,0,.2],'#1769aa']].forEach(([name,p,col])=>{line([[0,0,0],p],col,2);let[x,y]=project(p,W,H);ctx.fillStyle=col;ctx.fillText(name,x+5,y)});ctx.fillStyle='#59697b';ctx.fillText('固定根坐标系 · 等比例正交投影 · 坐标轴长度 0.2 m',15,H-15);
el('time').textContent=D.time[frame].toFixed(3)+' / '+D.time.at(-1).toFixed(3)+' s';
el('readout').replaceChildren();[['COM',D.com[frame]],...D.monitor.map(i=>[D.links[i],D.frames[frame][i]])].forEach(([name,p])=>{let div=document.createElement('div');div.className='metric';div.textContent=name+': ('+p.map(v=>v.toFixed(3)).join(', ')+') m';el('readout').appendChild(div)});
}
function plot(id,title,series,labels,bounds=[]){let W=600,H=255,L=65,R=20,T=30,B=40;let vals=series.flat().concat(bounds),min=Math.min(...vals),max=Math.max(...vals),pad=Math.max((max-min)*.1,1e-5);min-=pad;max+=pad;const x=t=>L+t/D.time.at(-1)*(W-L-R),y=v=>T+(max-v)/(max-min)*(H-T-B);let s=`<svg class="plot" viewBox="0 0 ${W} ${H}" aria-label="${title}"><text x="${L}" y="18" font-size="14">${title}</text>`;for(let k=0;k<5;k++){let v=min+(max-min)*k/4;s+=`<path d="M${L} ${y(v)}H${W-R}" stroke="#e0e6ed"/><text x="2" y="${y(v)+4}" font-size="11">${v.toPrecision(3)}</text>`}bounds.forEach(v=>s+=`<path d="M${L} ${y(v)}H${W-R}" stroke="#657385" stroke-dasharray="6 4"/>`);series.forEach((values,k)=>{s+=`<polyline points="${values.map((v,i)=>x(D.time[i])+','+y(v)).join(' ')}" fill="none" stroke="${C[k%4]}" stroke-width="2"/>`});s+=`<path d="M${x(D.time[frame])} ${T}V${H-B}" stroke="#e88a17"/><text x="${L}" y="${H-12}" font-size="12">0 s</text><text x="${W-60}" y="${H-12}" font-size="12">${D.time.at(-1)} s</text></svg>`;el(id).innerHTML=s}
function updatePlots(){let i=+el('joint').value,r=D.summary[i],u=D.units[i],force=r.effort_unit;plot('position','位置 / '+u,[D.q.map(v=>v[i])],['q'],D.position_limits[i]||[]);plot('velocity','速度 / '+u+'/s',[D.qd.map(v=>v[i])],['qd'],r.velocity_limit?[-r.velocity_limit,r.velocity_limit]:[]);plot('acceleration','加速度 / '+u+'/s²',[D.qdd.map(v=>v[i])],['qdd']);plot('effort','关节力矩或力 / '+force,[D.tau.map(v=>v[i])],['effort'],r.effort_limit?[-r.effort_limit,r.effort_limit]:[]);el('jointState').textContent='当前需求 '+D.tau[frame][i].toFixed(3)+' '+force+' ｜ 峰值 '+r.peak_abs_effort.toFixed(3)+' '+force;el('positionSvg').href='figures/joint_'+String(i).padStart(2,'0')+'_position.svg';el('effortSvg').href='figures/joint_'+String(i).padStart(2,'0')+'_effort.svg';plot('comPlot','整机质心坐标 / m',[0,1,2].map(k=>D.com.map(p=>p[k])),['X','Y','Z'])}
function update(){el('seek').value=frame;draw();updatePlots()}
el('supportNote').textContent=D.polygon?'已提供支撑多边形：动画中绿色边界显示输入的支撑区域，橙点显示质心投影。结果属于准静态筛查。':'尚未提供真实支撑多边形，支撑区域及质心投影检查未评估。网格高度仅为显示参考，不表示地面接触模型。';
el('play').onclick=()=>{if(!playing&&frame===D.time.length-1){frame=0;clockTime=0}playing=!playing;last=null;el('play').textContent=playing?'暂停':'播放'};
el('seek').oninput=()=>{frame=+el('seek').value;clockTime=D.time[frame];update()};el('joint').onchange=updatePlots;['view','trails','labels'].forEach(id=>el(id).onchange=draw);el('reset').onclick=()=>{yaw=-.65;pitch=.35;zoom=1;draw()};canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)};canvas.onpointermove=e=>{if(drag&&el('view').value==='3d'){yaw+=(e.clientX-drag[0])*.009;pitch=Math.max(-1.4,Math.min(1.4,pitch+(e.clientY-drag[1])*.009));drag=[e.clientX,e.clientY];draw()}};canvas.onpointerup=canvas.onpointercancel=()=>drag=null;canvas.onwheel=e=>{e.preventDefault();zoom=Math.max(.3,Math.min(5,zoom*Math.exp(-e.deltaY*.001)));draw()};window.onresize=draw;
function tick(now){if(playing){if(last!==null)clockTime=Math.min(D.time.at(-1),clockTime+(now-last)/1000*(+el('speed').value));let next=0;while(next<D.time.length-1&&D.time[next]<clockTime)next++;if(next!==frame){frame=next;update()}if(clockTime>=D.time.at(-1)){playing=false;el('play').textContent='播放'}}last=now;requestAnimationFrame(tick)}update();requestAnimationFrame(tick);
</script></html>'''
    # Escape JSON against terminating the inert script element through a URDF name.
    serialized=json.dumps(data,ensure_ascii=False,allow_nan=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    (out/'visualization.html').write_text(template.replace('__DATA__',serialized),encoding='utf-8')


def run(robot, cfg, out):
    t,q,qd,qdd=make_trajectory(robot,cfg)
    links=cfg['monitor_links']
    if not links or len(set(links)) != len(links) or any(l not in robot.links for l in links):
        raise ValueError('monitor_links must be nonempty unique URDF link names')
    g=vec(cfg['gravity_m_s2'])
    if abs(g[0])+abs(g[1])>1e-10 or g[2]>=0:
        raise ValueError('This report assumes root Z points up and gravity is negative Z')
    overrides=cfg['capacity_overrides']
    if set(overrides)-set(robot.names):
        raise ValueError('Unknown joint in capacity_overrides')
    for values in overrides.values():
        if set(values)-{'effort','velocity'} or any(not math.isfinite(float(v)) or float(v)<=0 for v in values.values()):
            raise ValueError('Capacity overrides must be positive effort/velocity values')
    required_margin=float(cfg['support_margin_m'])
    if not math.isfinite(required_margin) or required_margin<0:
        raise ValueError('support_margin_m must be nonnegative')
    results=[robot.evaluate(a,b,c,cfg['payloads'],g) for a,b,c in zip(q,qd,qdd)]
    tau=np.array([r['tau'] for r in results]); com=np.array([r['com'] for r in results])
    force=np.array([r['base_force'] for r in results]);moment=np.array([r['base_moment'] for r in results])
    positions={l:np.array([r['states'][l][0] for r in results]) for l in links}
    rotations={l:np.array([r['states'][l][1].flatten() for r in results]) for l in links}
    margins=None
    if cfg['support_polygon_xy_m'] is not None:
        margins=np.array([polygon_margin(c[:2],cfg['support_polygon_xy_m']) for c in com])
    joint_report=[]
    for i,j in enumerate(robot.active):
        limit=j['limits']; caps={**limit,**overrides.get(j['name'],{})}
        effort,velocity=caps.get('effort',0),caps.get('velocity',0)
        bounds=j['kind'] != 'continuous'
        pos_ok=bool(np.all(q[:,i]>=limit['lower']-1e-9) and np.all(q[:,i]<=limit['upper']+1e-9)) if bounds else True
        peak=float(np.max(abs(tau[:,i])))
        rms=float(np.sqrt(np.trapezoid(tau[:,i]**2,t)/t[-1])) if hasattr(np,'trapezoid') else float(np.sqrt(np.trapz(tau[:,i]**2,t)/t[-1]))
        joint_report.append(dict(joint=j['name'],effort_unit='N' if j['kind']=='prismatic' else 'N m',
            position_min=float(q[:,i].min()),position_max=float(q[:,i].max()),position_pass=pos_ok,
            peak_abs_velocity=float(np.max(abs(qd[:,i]))),velocity_limit=velocity if velocity>0 else None,
            velocity_pass=bool(np.max(abs(qd[:,i]))<=velocity+1e-9) if velocity>0 else None,
            peak_abs_effort=peak,rms_effort=rms,effort_limit=effort if effort>0 else None,
            effort_pass=bool(peak<=effort+1e-9) if effort>0 else None,
            effort_limit_source='config override' if 'effort' in overrides.get(j['name'],{}) else 'URDF screening value'))
    limits=['Fixed root; base wrench is the required support wrench, not solved wheel contact forces.',
            'URDF effort screening does not validate continuous/peak duration, thermal, gearbox or torque-speed capacity.',
            'COM projection is quasi-static geometric screening only; dynamic stability is NOT evaluated.',
            'No collision/clearance, end-effector task reachability/IK, perception or navigation validation.',
            'Payloads are point masses; finite payload inertia, gripper and tool offset need explicit modelling.',
            'Sampled checks do not guarantee continuous-time feasibility; refine sampling near boundaries.',
            'Mechanical work is not battery energy; no efficiency, electronics power or regeneration model.']
    checks=[v for r in joint_report for k,v in r.items() if k.endswith('_pass')]
    screening='FAIL' if any(v is False for v in checks) else 'INCOMPLETE' if any(v is None for v in checks) else 'PASS_FOR_MODEL_LIMITS_ONLY'
    report=dict(case_name=cfg['case_name'],urdf=str(robot.path.resolve()),
        urdf_sha256=hashlib.sha256(robot.path.read_bytes()).hexdigest(),
        model_links=len(robot.links),active_joints=len(robot.active),root_frame=robot.root,
        urdf_mass_kg=robot.mass,total_mass_with_payload_kg=results[0]['mass'],duration_s=float(t[-1]),samples=len(t),
        motion='neutral/static hold' if np.allclose(q[0],q[-1]) else 'user-specified quintic joint interpolation',
        model_limit_screening=screening,overall_design_validation='NOT_ESTABLISHED',
        quasi_static_com_projection=dict(status='NOT_EVALUATED' if margins is None else 'PASS' if margins.min()>=required_margin else 'FAIL',
            minimum_edge_margin_m=None if margins is None else float(margins.min()),required_margin_m=required_margin),
        warnings=robot.warnings,limitations=limits,joints=joint_report)
    out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    (out/'config_used.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    joint_cols=[]
    for i,n in enumerate(robot.names):
        unit='m' if robot.active[i]['kind']=='prismatic' else 'rad'
        joint_cols += [f'{n}.q_{unit}',f'{n}.qd_{unit}_s',f'{n}.qdd_{unit}_s2',f'{n}.effort']
    rows=[]
    for k in range(len(t)):
        row=[float(t[k])]+[float(v) for i in range(len(robot.names)) for v in (q[k,i],qd[k,i],qdd[k,i],tau[k,i])]
        rows.append(row)
    write_csv(out/'joint_timeseries.csv',['time_s']+joint_cols,rows)
    headers=['time_s','com_x_m','com_y_m','com_z_m','base_fx_N','base_fy_N','base_fz_N','base_mx_Nm','base_my_Nm','base_mz_Nm']
    pose=np.column_stack([t,com,force,moment])
    for l in links:
        headers += [f'{l}.{a}_m' for a in 'xyz']+[f'{l}.R{i}{j}' for i in range(3) for j in range(3)]
        pose=np.column_stack([pose,positions[l],rotations[l]])
    if margins is not None:
        headers += ['quasi_static_edge_margin_m'];pose=np.column_stack([pose,margins])
    write_csv(out/'body_timeseries.csv',headers,pose)
    write_csv(out/'joint_summary.csv',list(joint_report[0]),[list(r.values()) for r in joint_report])
    content=['<!doctype html><meta charset="utf-8"><title>Robot verification</title><style>body{font:16px system-ui;max-width:1050px;margin:35px auto;padding:15px}svg{width:100%}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:6px;text-align:left}</style>',
             '<h1>Fixed-base robot verification</h1><p><a href="visualization.html">Open interactive trajectory and verification dashboard</a></p>',f'<p>Case: {html.escape(str(cfg["case_name"]))}. Model screening: <b>{screening}</b>. Overall design validation: <b>NOT ESTABLISHED</b>.</p>',
             '<p>Root frame: '+html.escape(robot.root)+f'. URDF mass: {robot.mass:.5f} kg. Total with payloads: {results[0]["mass"]:.5f} kg.</p>',
             '<h2>Scope and missing checks</h2><ul>'+''.join('<li>'+html.escape(s)+'</li>' for s in limits+robot.warnings)+'</ul>',
             chart('Whole-body COM (m)',t,com,['x','y','z'])]
    for l in links:
        content.append(chart(f'{l} origin position (m)',t,positions[l],['x','y','z']))
    if margins is not None:
        content.append(chart('Quasi-static COM edge margin (m)',t,margins,['signed margin']))
    content.append('<h2>Joint demand</h2>')
    for i,j in enumerate(robot.active):
        content.append(chart(j['name']+' effort ('+joint_report[i]['effort_unit']+')',t,tau[:,i],['required effort']))
    (out/'report.html').write_text(''.join(content),encoding='utf-8')
    write_visualizations(robot, cfg, out, t, q, qd, qdd, tau, com, results, joint_report)
    print(f"Visualization: {(out/'visualization.html').resolve()}")
    print(f"URDF mass: {robot.mass:.5f} kg; with payloads: {results[0]['mass']:.5f} kg")
    print(f'Model-limit screening: {screening}; overall design validation: NOT ESTABLISHED')
    print(f'Report: {(out/"report.html").resolve()}')
    return report


def self_test():
    """Independent analytic pendulum + energy/Jacobian and inertia consistency tests."""
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'pendulum.urdf'
        p.write_text('''<robot name="test"><link name="base"><inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial></link><link name="bob"><inertial><origin xyz="1 0 0"/><mass value="2"/><inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/></inertial></link><joint name="hinge" type="revolute"><parent link="base"/><child link="bob"/><axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="100" velocity="10"/></joint></robot>''')
        r=Robot(p);q=np.array([.4]);qd=np.array([.8]);qdd=np.array([1.2])
        got=r.evaluate(q,qd,qdd)['tau'][0]
        expected=2.1*1.2-2*9.80665*math.cos(.4)
        np.testing.assert_allclose(got,expected,atol=1e-10)
        payload=[dict(link='bob',mass_kg=3,com_xyz_m=[1,0,0])]
        got=r.evaluate(q,qd,qdd,payload)['tau'][0]
        np.testing.assert_allclose(got,5.1*1.2-5*9.80665*math.cos(.4),atol=1e-10)
        # A prismatic actuator lifts its own body and the attached payload.
        p.write_text(p.read_text().replace('type="revolute"','type="prismatic"').replace('xyz="0 1 0"','xyz="0 0 1"'))
        pr=Robot(p)
        np.testing.assert_allclose(pr.evaluate([.4],[.8],[1.2],payload)['tau'],[5*(9.80665+1.2)],atol=1e-10)
    r=Robot(DEFAULT_URDF);n=len(r.names)
    q=np.array([.1 if j['kind']=='continuous' else .3*j['limits']['lower']+.7*j['limits']['upper'] for j in r.active])
    qd=np.linspace(-.2,.2,n);eps=1e-6;g=np.array([0,0,-9.80665])
    # Gravitational generalized forces equal finite-difference potential gradient.
    numerical=[]
    for i in range(n):
        delta=np.zeros(n);delta[i]=eps
        a=r.evaluate(q+delta);b=r.evaluate(q-delta)
        numerical.append((-a['mass']*g@a['com']+b['mass']*g@b['com'])/(2*eps))
    np.testing.assert_allclose(r.evaluate(q)['tau'],numerical,atol=2e-6,rtol=1e-6)
    # Forward recursion COM accelerations vs finite differences of poses.
    s=r.state(q,qd)[0]
    for name in r.links:
        def cg(st):return st[name][0]+st[name][1]@r.links[name]['c']
        step=1e-4
        ap=r.state(q+step*qd)[0];am=r.state(q-step*qd)[0]
        pp,rr,w,al,ac=s[name];c=rr@r.links[name]['c']
        expected=(cg(ap)-2*cg(s)+cg(am))/step**2
        np.testing.assert_allclose(ac+np.cross(al,c)+np.cross(w,np.cross(w,c)),expected,atol=2e-6)
    # Whole-tree mass matrix symmetry and positive definiteness.
    zero=np.zeros(n)
    M=np.column_stack([r.evaluate(q,zero,np.eye(n)[i],gravity=[0,0,0])['tau'] for i in range(n)])
    np.testing.assert_allclose(M,M.T,atol=1e-10)
    assert np.linalg.eigvalsh(M).min()>0
    # Velocity-dependent terms must satisfy the kinetic-energy power identity.
    def mass_matrix(at):
        return np.column_stack([r.evaluate(at,zero,np.eye(n)[i],gravity=[0,0,0])['tau'] for i in range(n)])
    step=1e-5
    mdot=(mass_matrix(q+step*qd)-mass_matrix(q-step*qd))/(2*step)
    bias=r.evaluate(q,qd,zero,gravity=[0,0,0])['tau']
    np.testing.assert_allclose(qd@bias,.5*qd@mdot@qd,atol=1e-8,rtol=1e-6)
    assert polygon_margin([0,0],[[-1,-1],[1,-1],[1,1],[-1,1]])==1
    assert polygon_margin([2,0],[[-1,-1],[1,-1],[1,1],[-1,1]])==-1
    print('PASS: analytic revolute/payload/prismatic dynamics, gravity gradient, COM acceleration, mass matrix and polygon tests')


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--urdf',type=Path,default=DEFAULT_URDF)
    case_group=parser.add_mutually_exclusive_group()
    case_group.add_argument('--config',type=Path)
    case_group.add_argument('--demo-motion',action='store_true',help='synthetic leg movement + 3 kg point payload; NOT validated sit/stand')
    parser.add_argument('--init-config',type=Path,help='write editable JSON template and exit (never overwrites)')
    parser.add_argument('--output',type=Path,default=HERE/'verification_output')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        self_test();return
    robot=Robot(args.urdf)
    cfg=config_template(robot)
    if args.init_config:
        with args.init_config.open('x',encoding='utf-8') as f:json.dump(cfg,f,ensure_ascii=False,indent=2)
        print(f'Configuration template: {args.init_config.resolve()}');return
    if args.demo_motion:
        required={'leg_joint_1','leg_joint_2','leg_joint_3'}
        if not required.issubset(robot.names) or 'left_arm_link_7' not in robot.links:
            raise ValueError('Demo requires the supplied VC robot joint/link names')
        cfg['case_name']='DEMO ONLY - synthetic motion, not validated sit/stand'
        cfg['end_q'].update(leg_joint_1=.3,leg_joint_2=-.6,leg_joint_3=.3)
        cfg['payloads']=[dict(link='left_arm_link_7',mass_kg=3,com_xyz_m=[.1,0,0])]
    if args.config:
        supplied=json.loads(args.config.read_text(encoding='utf-8'))
        unknown=set(supplied)-set(cfg)
        if unknown:raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
        cfg.update(supplied)
    run(robot,cfg,args.output)


if __name__=='__main__':
    try:
        main()
    except (ValueError,KeyError,TypeError,OSError,ET.ParseError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr);sys.exit(2)
