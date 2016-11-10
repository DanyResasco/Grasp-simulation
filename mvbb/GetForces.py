#!/usr/bin/env python
from klampt import *
from klampt.vis.glrobotprogram import * #Per il simulatore
from klampt.model import collide
from moving_base_control import set_moving_base_xform, get_moving_base_xform
import numpy as np
from IPython import embed 
from klampt.math import se3,so3

def get_contact_forces_and_jacobians(robot,world,sim):
    """
    Returns a force contact vector 1x(6*n_contacts)
    and a contact jacobian matrix 6*n_contactsxn.
    Contact forces are considered to be applied at the link origin
    """
    # n_contacts = 0  # one contact per link
    maxid = world.numIDs()
    # print"maxid",maxid
    f_l = dict()
    # contacts_l_id_j = None

    for l_id in range(robot.numLinks()):
        # link = robot.link(robot.driver(i).getName())
    #     u_to_l.append(robot.link(i).getID())
    # for l_id in u_to_l:
        link_in_contact = robot.link(l_id).getID()
        # contacts_per_link = 0
        for j in xrange(maxid):  # TODO compute just one contact per link
            contacts_l_id_j = len(sim.getContacts(link_in_contact, j))
            # print"contacts_l_id_j",contacts_l_id_j
            # contacts_per_link += contacts_l_id_j
            if contacts_l_id_j > 0:
                for k in range(0,contacts_l_id_j):
                    # print"k",k
                    if not f_l.has_key(l_id):
                        # print"dentro il if not e prima di contactForce"
                        f_l[l_id] = sim.contactForce(link_in_contact, k)
                        # print"dentro il if not e dopo di contactForce"
                    else:
                        # print"dentro else e prima di contactForce"
                        f_l[l_id] = vectorops.add(f_l[l_id], sim.contactForce(link_in_contact, k))
                        # print"dentro else e dopo di contactForce"
                if contacts_l_id_j > 1:
                    f_l[l_id] = vectorops.div(f_l[l_id], contacts_l_id_j)
                print"nome",robot.link(l_id).getName(),"force", f_l[l_id]
    return (f_l)
    # colliding = []
    # sim.enableContactFeedbackAll()
    # n = sim.world.numIDs()
    # colliding = range(n)
    # Fc = []
    # for i,id1 in enumerate(colliding):
    #     for j in range(i+1,len(colliding)):
    #         id2 = colliding[j]
    #         if sim.hadContact(id1,id2):
    #             clist = sim.getContacts(id1,id2);
    #             f = sim.contactForce(id1,id2)
    #             m = sim.contactTorque(id1,id2)
    #             pavg = [0.0]*3
    #             navg = [0.0]*3
    #             for c in clist:
    #                pavg = vectorops.add(pavg,c[0:3])
    #                navg = vectorops.add(navg,c[3:6])
    #             if len(clist) > 0:
    #                pavg = vectorops.div(pavg,len(clist))
    #                navg = vectorops.div(navg,len(clist))
    #             body1 = world.getName(id1)
    #             body2 = world.getName(id2)
    #             cvalues = [sim.getTime(),body1,body2,len(clist)]
    #             cvalues += pavg
    #             cvalues += navg
    #             cvalues += f
    #             cvalues += m
    #             Fc = [v for v in cvalues[v]]
    # return Fc