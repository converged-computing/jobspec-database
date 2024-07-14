args <- commandArgs(trailingOnly = TRUE)
cnvfile <- args[1] #CNV file to load. Must have filename and plateID variables!
study <- args[2] #Name of study (for naming files)


#cnvfile <- 'ressler_gtp_cnv.csv'
#path <- '/home/amaihofer/gtpc/intensities'
#studyname <- 'gtpc'


dat <- read.csv(cnvfile,stringsAsFactors=F,header=T)

#Assuming that correct order has been given already
#dat <- dat[order(dat$plateID),] #Order by plate

#Add plates together until N ~ 200
#Count N per plate
platelist <- table(dat$plateID)

#List object that will say plate grouping. List is the length of the number of plates, as at maximum each plate would be done separately in iPattern
plateset <- vector("list", length(platelist))

#Aggregate plates by adding to 
#Need to fix problem that dataset will not be definied
platecall <- 1
realincrement=1
for (i in 1:length(platelist)) #Have to set loop based on value of increment
{
  print(realincrement)
  #If we've done every assignment, just break the loop now
  if(realincrement > length(platelist)) {  break }
  curlength <- sum(platelist[realincrement])
  curnames <- names(platelist[realincrement])

  dev1 <- abs(200 - curlength)
  
   #We're going to add samples until it stops making sense to do so
   #i.e. if based on closeness to value 200, it makes sense to add sampels still, then add them, and furthermore, see if adding even more samples is beneficial
   dev0<-dev1
   dev2<-dev1
   increment=1
   while(dev0 >= dev1 ){
    lengthprev <- sum(platelist[realincrement:(realincrement+increment-1)])
    namesprev <- names(platelist[realincrement:(realincrement+increment-1)])   
    lengthcur <- sum(platelist[realincrement:(realincrement+increment)])
    namescur <- names(platelist[realincrement:(realincrement+increment)])
    lengthplus <- sum(platelist[realincrement:(realincrement+increment+1)])
    namesplus <- names(platelist[realincrement:(realincrement+increment+1)])
    dev0 <- abs(200 - lengthprev)
    dev1 <- abs(200 - lengthcur)
    dev2 <- abs(200 - lengthplus)
    #there are probably other circumstances when this would be NA, but as it is, it's a good indicator of when we've run out of observations and can terminate.
    if(is.na(lengthcur))
    {
     print("End of dataset!")
     dev1=1000
    }
    increment=increment+1
   }
   if (dev0 <= dev1)
   {
    plateset[[platecall]] <- namesprev
    realincrement=realincrement+increment-1
    print("Use dev0")
   } else   if (dev1 <= dev2)
   {
    plateset[[platecall]] <- namescur
    realincrement=realincrement+increment
    print("Use dev1")
   } else   if(dev1 > dev2)
   {
    plateset[[platecall]] <- namesplus
    realincrement=realincrement+increment+1
    print("Use dev2")
   } 

  # print(realincrement)
   #print(platecall)
   #Reset counters
   platecall=platecall+1
   increment=1
}

#Now we have achieved a list of plates. Export all elements of data for plates
plateset[sapply(plateset, is.null)] <- NULL
 
for (assignments in 1:length(plateset))
{
 dexport <- subset(dat,plateID %in% plateset[[assignments]], select=fullpath)
 write.table(dexport,paste(study,"_",assignments,sep=""),row.names=F,quote=F,col.names=F)
 }
 
 

 