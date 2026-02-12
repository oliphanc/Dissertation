// Simcenter STAR-CCM+ macro: updateStartingConditions.java
// Written by Chase Oliphant, 9/20/2024
// Updates the rotation rate and starting mass flow rate of the simulation
// Rotation rate is updated to be the environment variable "SPEEDLINERPM"
// Mass flow rate is updated to maintain flow coefficient (assuming constant density)

package macro;

import java.util.*;

import star.common.*;
import star.base.neo.*;
import star.base.report.*;
import star.flow.*;
import star.vis.*;
import star.motion.*;
import java.io.PrintWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.File;
import java.nio.file.Paths;
import java.nio.file.Path;
import java.nio.file.Files;


public class updateStartingConditions extends StarMacro {

  public void execute() {
	try {
		execute0();
	}
	catch(Exception except) {
		Simulation sim = getActiveSimulation();
		sim.println("ERROR: Unable to set simulation RPM.");
		sim.println(except);
		System.exit(10);
	}
  }

  private void execute0() {
	double N = Double.valueOf(System.getenv("speed"));
	double mdot = getMdot();
	double oldN = getRPM();
	
	Simulation sim = getActiveSimulation();
	sim.println(oldN);
	mdot = mdot * N/oldN;
	setMdot(mdot);
	
	RotatingMotion rotating = ((RotatingMotion) sim.get(MotionManager.class).getObject("Rotation"));
	RotationRate rotationRate = ((RotationRate) rotating.getRotationSpecification());
	rotationRate.getRotationRate().setValue(N);
	
	String simName = sim.getPresentationName();
	sim.saveState(simName + ".sim");
	continueSim(4000);
	sim.saveState(simName + ".sim");
	saveSimForSpeedline(sim);
  } 
  
  private double getMdot() {
	
	Simulation sim = getActiveSimulation();
	Collection<Region> regions = sim.getRegionManager().getRegions();
	Region fluidRegion = (Region) regions.iterator().next();
	Boundary boundary = fluidRegion.getBoundaryManager().getBoundary("Outlet");
	MassFlowRateProfile massFlowRateProfile_0 = boundary.getValues().get(MassFlowRateProfile.class);
	double mdot = massFlowRateProfile_0.getMethod(ConstantScalarProfileMethod.class).getQuantity().getValue();
	
	if (mdot < 0){
		mdot = -mdot;
	}
	
	return mdot;
  }

  private void setMdot(double mdot) {

	Simulation sim = getActiveSimulation();
	Collection<Region> regions = sim.getRegionManager().getRegions();
	Region fluidRegion = (Region) regions.iterator().next();
	Boundary boundary = fluidRegion.getBoundaryManager().getBoundary("Outlet");
	MassFlowRateProfile massFlowRateProfile_0 = boundary.getValues().get(MassFlowRateProfile.class);
	massFlowRateProfile_0.getMethod(ConstantScalarProfileMethod.class).getQuantity().setValue(mdot);
  }

  private double getRPM() {
	Simulation sim = getActiveSimulation();
	
	RotatingMotion rotating = ((RotatingMotion) sim.get(MotionManager.class).getObject("Rotation"));
	RotationRate rotationRate = ((RotationRate) rotating.getRotationSpecification());
	return rotationRate.getRotationRate().getRawValue();
  }
 
  private void continueSim(int steps) {
	
	Simulation sim = getActiveSimulation();
	int iter = sim.getSimulationIterator().getCurrentIteration();
	steps = steps + iter;
	
	StepStoppingCriterion stepStoppingCriterion_0 = 
		((StepStoppingCriterion) sim.getSolverStoppingCriterionManager().getSolverStoppingCriterion("Maximum Steps"));

	stepStoppingCriterion_0.setMaximumNumberSteps(steps);
	sim.getSimulationIterator().run();
  }

  private void saveSimForSpeedline(Simulation sim) {
	
	String N = System.getenv("speed");
	Path speedlinePath = createDirectory(N);
	double mdot = getMdot();
	sim.saveState(speedlinePath.toString() + "/" + String.format("%.5f", mdot) + ".sim");
	PrintWriter writer = null; 
	try {
		writer = new PrintWriter(new FileWriter(".SIMNAME", false));
		writer.println(sim.getPresentationName() + ".sim");
	} catch (IOException e) {
		e.printStackTrace();
	} finally {
		if (writer != null) {
			writer.close();
		}
	}
  }

 private Path createDirectory(String name) {
	  
	Simulation sim = getActiveSimulation();
	String simPathString = sim.getSessionPath();
	Path simPath = Paths.get(simPathString);
	String simDirectory = simPath.getParent().toString();
	Path newDirectory = Paths.get(simDirectory + "/" + name);
	try {
		Files.createDirectories(newDirectory);
	} catch (IOException e) {
		e.printStackTrace(); // Handle file I/O exceptions
	}
	
	return newDirectory;
  } 

}
