// Simcenter STAR-CCM+ macro: simulationCheck.java
// Written by Chase Oliphant, 9/20/2024
// Checks that the simulation meets all of the requirements to run properly

package macro;

import java.util.*;

import star.common.*;
import star.base.report.*;
import star.base.neo.*;
import star.vis.*;
import star.flow.*;
import star.motion.*;

public class simulationCheck extends StarMacro {
  private static boolean noError = true;
  public void execute() {
    execute0();
  } 

  private void execute0() {
    Simulation sim = getActiveSimulation();
	
	Collection<Region> regions = sim.getRegionManager().getRegions();
	Region fluidRegion = (Region) regions.iterator().next();
	try {
		Boundary bound = fluidRegion.getBoundaryManager().getBoundary("Outlet");
	} catch (Exception error) {
		sim.println("***ERROR: No boundary named 'Outlet'. Please check simulation***");
		noError = false;
	}
	
	try {
		MassFlowReport massFlowReport_0 = ((MassFlowReport) sim.getReportManager().getReport("Mass Flow P2"));
	} catch (Exception error) {
		sim.println("***ERROR: No Mass Flow P2 Report***");
		sim.println("Attempting to create...");
		noError = false;
		if(creatMassFlowP2Report()) {
			sim.println("Report created!");
			noError = true;
		}
	}
	
	if (noError) {
		sim.println("\n\n***No errors detected. Simulation is ready to run***\n\n");
	}
  }
  
  private boolean creatMassFlowP2Report() {
	Simulation sim = getActiveSimulation();
	MassFlowReport massFlowP2 = sim.getReportManager().createReport(MassFlowReport.class);

    massFlowP2.setPresentationName("Mass Flow P2");
    massFlowP2.getParts().setQuery(null);
	
	try {
		CylinderSection impellerOutlet = ((CylinderSection) 
			sim.getPartManager().getObject("Impeller Outlet Section"));
		massFlowP2.getParts().setObjects(impellerOutlet);
		sim.saveState(sim.getSessionPath());
	} catch (Exception e) {
		sim.println("\n\nUnable to create report. Please ensure that the Impeller Outlet Section exists.\n\n");
		sim.println(e);
		return false;
	}
	
	return true;
    
  }
}