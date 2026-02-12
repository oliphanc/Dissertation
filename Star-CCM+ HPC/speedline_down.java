// STAR-CCM+ macro: test.java
// Written by STAR-CCM+ 15.02.009

// A basic macro to run the entire speedline for a radial impeller with a volute.
// All run points must be set by the user in the "execute function"


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


public class speedline_down extends StarMacro {
	
  private static String N = System.getenv("SPEEDLINERPM");
  private static String speedlineDirectory;
  private static double mdot = 0.0;
  private static double dM = -0.1;  //Percent of starting mass flow rate
 
  public void execute() {
	  
	start();
	
	runMassOutlet(4000);
	runMassOutlet(4000);
	runMassOutlet(4000);
	runMassOutlet(4000);
	exitSuccess();
  }
  
  private void start() {
	Simulation sim = getActiveSimulation();
	
	speedlineDirectory = createDirectory(N).toString();
	
	getMdot();
	dM = dM * mdot;
	sim.saveState(speedlineDirectory + "/" + String.format("%.5f", mdot) + ".sim");
  }

  
  private void runMassOutlet(int steps) {
	
	Simulation sim = getActiveSimulation();

	mdot = mdot + dM;

    Collection<Region> regions = sim.getRegionManager().getRegions();
	Region fluidRegion = (Region) regions.iterator().next();

    Boundary boundary = fluidRegion.getBoundaryManager().getBoundary("Outlet");
    MassFlowRateProfile massFlowRateProfile_0 = boundary.getValues().get(MassFlowRateProfile.class);
    massFlowRateProfile_0.getMethod(ConstantScalarProfileMethod.class).getQuantity().setValue(mdot);
	
    sim.saveState(speedlineDirectory + "/" + String.format("%.5f", mdot) + ".sim");
    continueSim(steps);
	sim.saveState(speedlineDirectory + "/" + String.format("%.5f", mdot) + ".sim");

	exportTables();
	exportReports();
	exportMonitors();
	
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
  
  private void exportMonitors() {
 
    Simulation sim = getActiveSimulation();
    Collection <Monitor> monitors = sim.getMonitorManager().getMonitors();
    Iterator monitorIt = monitors.iterator();
	
	
	Path monitorDirectory = createDirectory("Monitors");
	Path monitorSubDirectory = Paths.get(monitorDirectory.toString() + "/" + String.format("%.5f", mdot));
	
	try {
		Files.createDirectories(monitorSubDirectory);
	} catch (IOException e) {
		e.printStackTrace(); // Handle file I/O exceptions
	}
	
	while (monitorIt.hasNext()) {
		Monitor monitor = (Monitor) monitorIt.next();
		String monitorTitle = monitor.getPresentationName();
		monitorTitle = monitorTitle.replace("/", "_");
		String fileName = monitorSubDirectory.toString() + "/" + monitorTitle + ".csv";
		monitor.export(fileName);
	}
    
  }
  
  
   private void getMdot() {
	Simulation sim = getActiveSimulation();

	Collection<Region> regions = sim.getRegionManager().getRegions();
	Region fluidRegion = (Region) regions.iterator().next();
	Boundary boundary = fluidRegion.getBoundaryManager().getBoundary("Outlet");
	MassFlowRateProfile massFlowRateProfile_0 = boundary.getValues().get(MassFlowRateProfile.class);
	
	mdot = massFlowRateProfile_0.getMethod(ConstantScalarProfileMethod.class).getQuantity().getRawValue();
  }
  
  
  private void exportTables() {

    Simulation sim = getActiveSimulation();
	
	Path inletDirectory = createDirectory("Inlet Flow Fields");
	Path outletDirectory = createDirectory("Outlet Flow Fields");
	
    XyzInternalTable outletTable = 
		((XyzInternalTable) sim.getTableManager().getTable("Impeller Outlet Field"));
    outletTable.extract();
    outletTable.export(outletDirectory.toString() + 
		"/outlet@" + String.format("%.5f", mdot) + ".csv", ",");  
    
    XyzInternalTable inletTable = 
      ((XyzInternalTable) sim.getTableManager().getTable("Impeller Inlet Field"));
    inletTable.extract();
    inletTable.export(inletDirectory.toString() + 
		"/inlet@" + String.format("%.5f", mdot) + ".csv", ","); 
    
  }
  
  
  private void exportReports() {
	  
	Simulation sim = getActiveSimulation();
    Collection <Report> reps = sim.getReportManager().getObjects();
    Iterator repIt = reps.iterator();
	String simName = sim.getPresentationName();
	Path reportDirectory = createDirectory("Reports");
	
	String filepath = reportDirectory.toString() + 
		"/Reports@" + String.format("%.5f", mdot) + ".txt";
	PrintWriter writer = null;
	try {
		writer = new PrintWriter(new FileWriter(filepath, false));
		// Print the report to the PrintWriter
		while (repIt.hasNext())
		{
		  Report rep = (Report) repIt.next();
		  double repValue = rep.getReportMonitorValue();
		  String repTitle = rep.getPresentationName();
		  String repUnit = rep.getUnits().getPresentationName();
		  writer.println(repTitle + "\t" + repValue + "\t" + repUnit);
		  //rep.printReport();
		  
		}
	} catch (IOException e) {
		e.printStackTrace(); // Handle file I/O exceptions
	} finally {
		if (writer != null) {
			writer.close(); // Ensure the writer is closed properly
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
  
  private void exitSuccess() {
	
	try {
		File success = new File(".SUCCESS");
		success.createNewFile();
	} catch (IOException e){
		e.printStackTrace();
	}
	
  }
  
}

